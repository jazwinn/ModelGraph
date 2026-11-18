#include "Graph/Compiler.h"

#include <map>

namespace mg
{
    void Program::Compile(const Graph &graph, const Analysis &analysis, BuildState &state)
    {
        if (!state.buildSubgraph)
        {
            // Nests (Group / Repeat) compile their subgraph with the same state,
            // so KV-cache slots and RoPE tables are shared model-wide.
            state.buildSubgraph = [&state](const std::string &name, const std::vector<TensorType> &inputs) -> std::shared_ptr<Program>
            {
                const auto it = state.doc.subgraphs.find(name);
                if (it == state.doc.subgraphs.end()) throw GraphError("subgraph '" + name + "' does not exist");
                const Analysis sub = Analyze(state.doc, it->second, state.registry, inputs);
                if (!sub.Ok()) throw GraphError("subgraph '" + name + "' has errors:\n" + sub.Summary());
                auto program = std::make_shared<Program>();
                program->Compile(it->second, sub, state);
                return program;
            };
        }
        const Variables vars = state.doc.NumericVariables();

        std::map<int, int> stepOf;   // node id -> step index
        std::map<int, SpecPtr> specs;
        std::map<int, int> ioPosition;
        for (const NodeRole role : {NodeRole::Input, NodeRole::Output})
        {
            const std::vector<int> ids = OrderedIo(graph, state.registry, role);
            for (size_t i = 0; i < ids.size(); ++i) ioPosition[ids[i]] = static_cast<int>(i);
        }
        for (const int id : analysis.order)
        {
            const Node &node = *graph.Find(id);
            SpecPtr spec = state.registry.Find(node.type);
            if (!spec) throw GraphError("unknown node type '" + node.type + "'");
            specs[id] = spec;

            Step step;
            step.name = node.name;
            step.type = node.type;
            step.role = spec->role;
            const NodePorts &ports = analysis.ports.at(id);
            step.numOutputs = ports.outputs.size();
            step.inputs.resize(ports.inputs.size());

            Attrs attrs(*spec, node.attrs, vars);
            if (spec->role != NodeRole::Op)
            {
                step.ioIndex = ioPosition.at(id);
            }
            else
            {
                BuildArgs args{node, attrs, analysis.inputs.at(id), analysis.outputs.at(id), state};
                if (!spec->build) throw GraphError("node type '" + node.type + "' cannot be executed");
                try
                {
                    step.kernel = spec->build(args);
                }
                catch (const std::exception &e)
                {
                    throw GraphError("building '" + node.name + "': " + e.what());
                }
                if (auto module = step.kernel->GetModule())
                {
                    this->RegisterSubmodule(node.name, module);
                }
            }
            stepOf[id] = static_cast<int>(steps.size());
            steps.push_back(std::move(step));
        }

        for (const Link &l : graph.links)
        {
            Step &to = steps[stepOf.at(l.to.node)];
            const int port = analysis.ports.at(l.to.node).InputIndex(l.to.port);
            to.inputs[port] = {stepOf.at(l.from.node), analysis.ports.at(l.from.node).OutputIndex(l.from.port)};
        }

        for (const Step &s : steps)
        {
            if (s.role != NodeRole::Output) continue;
            if (outputs.size() <= static_cast<size_t>(s.ioIndex)) outputs.resize(s.ioIndex + 1);
            outputs[s.ioIndex] = s.inputs.at(0);
        }

        // Late binding between nodes (tied weights).
        std::map<std::string, std::shared_ptr<nn::Module<Scalar>>> modules;
        for (const Step &s : steps)
        {
            if (s.kernel)
                if (auto m = s.kernel->GetModule()) modules[s.name] = m;
        }
        for (Step &s : steps)
        {
            if (!s.kernel) continue;
            s.kernel->Resolve([&](const std::string &name) -> std::shared_ptr<nn::Module<Scalar>>
            {
                const auto it = modules.find(name);
                if (it == modules.end()) throw GraphError("'" + s.name + "' refers to '" + name + "', which has no weights");
                return it->second;
            });
        }

        // Count consumers so intermediate tensors are freed as early as possible.
        m_uses.assign(steps.size(), {});
        for (size_t i = 0; i < steps.size(); ++i) m_uses[i].assign(steps[i].numOutputs, 0);
        for (const Step &s : steps)
            for (const Source &src : s.inputs)
                if (src.step >= 0) ++m_uses[src.step][src.port];
        for (const Source &src : outputs)
            if (src.step >= 0) ++m_uses[src.step][src.port];
    }

    std::vector<Value> Program::Run(RunContext &ctx, const std::vector<Value> &inputs)
    {
        std::vector<std::vector<Value>> values(steps.size());
        std::vector<std::vector<int>> uses = m_uses;

        auto take = [&](const Source &src) -> Value
        {
            if (src.step < 0) return std::monostate{};
            Value v = values[src.step][src.port];
            if (--uses[src.step][src.port] == 0) values[src.step][src.port] = std::monostate{};
            return v;
        };

        for (size_t i = 0; i < steps.size(); ++i)
        {
            const Step &s = steps[i];
            if (s.role == NodeRole::Input)
            {
                if (static_cast<size_t>(s.ioIndex) >= inputs.size())
                {
                    throw GraphError("'" + s.name + "' expects input " + std::to_string(s.ioIndex) + " but only " +
                                     std::to_string(inputs.size()) + " were given");
                }
                values[i] = {inputs[s.ioIndex]};
                continue;
            }
            if (s.role == NodeRole::Output) continue;

            std::vector<Value> in;
            in.reserve(s.inputs.size());
            for (const Source &src : s.inputs) in.push_back(take(src));
            try
            {
                values[i] = s.kernel->Run(ctx, in);
            }
            catch (const std::exception &e)
            {
                throw GraphError("'" + s.name + "' (" + s.type + "): " + e.what());
            }
            if (values[i].size() != s.numOutputs)
            {
                throw GraphError("'" + s.name + "' returned " + std::to_string(values[i].size()) + " values, expected " +
                                 std::to_string(s.numOutputs));
            }
        }

        std::vector<Value> result;
        for (const Source &src : outputs)
        {
            if (src.step < 0) throw GraphError("an output node is not connected");
            result.push_back(values[src.step][src.port]);
        }
        return result;
    }

    std::shared_ptr<GraphModel> GraphModel::Build(const Document &doc, const Registry &registry)
    {
        const Analysis analysis = Analyze(doc, doc.main, registry);
        if (!analysis.Ok()) throw GraphError("the graph has errors:\n" + analysis.Summary());

        int inputCount = 0;
        for (const Node &n : doc.main.nodes)
        {
            const SpecPtr spec = registry.Find(n.type);
            if (spec && spec->role == NodeRole::Input) ++inputCount;
        }
        if (inputCount != 1) throw GraphError("the main graph needs exactly one TokenIds input");
        if (analysis.results.empty()) throw GraphError("the main graph needs a Logits output");

        auto model = std::make_shared<GraphModel>();
        BuildState state{doc, registry, {}, {}, {}};
        model->Compile(doc.main, analysis, state);
        model->slots = state.slots;
        return model;
    }

    FTensor GraphModel::Forward(const ITensor &ids, GraphCache *cache)
    {
        if (ids.Rank() != 2) throw GraphError("token ids must be a [batch, seq] tensor");
        RunContext ctx;
        ctx.cache = cache;
        ctx.seqLen = ids.Shape()[1];
        ctx.startPos = cache ? cache->Length() : 0;
        ctx.device = m_device;
        if (cache && ctx.startPos + ctx.seqLen > cache->MaxLength())
        {
            throw GraphError("sequence exceeds the cache length");
        }

        const std::vector<Value> out = Run(ctx, {ids.GetDevice() == m_device ? ids : ids.To(m_device)});
        if (cache) cache->Advance(ctx.seqLen);
        return AsFloat(out.at(0), "model output");
    }

    GraphCache GraphModel::MakeCache(size_t batch, size_t maxSeq) const
    {
        return GraphCache(slots, batch, maxSeq, m_device);
    }

    void GraphModel::ToDevice(tensor::Device device)
    {
        To(device);
        m_device = device;
    }

    int64_t GraphModel::ParameterCount() const
    {
        int64_t total = 0;
        for (const auto &p : Parameters()) total += static_cast<int64_t>(p->Size());
        return total;
    }
}
