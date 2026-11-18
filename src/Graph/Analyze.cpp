#include "Graph/Analyze.h"

#include <algorithm>
#include <queue>
#include <set>

namespace mg
{
    namespace
    {
        constexpr int kMaxDepth = 16;
    }

    std::string Analysis::Summary() const
    {
        std::string text;
        for (const std::string &e : graphErrors) text += e + "\n";
        for (const auto &[id, e] : errors) text += "node " + std::to_string(id) + ": " + e + "\n";
        return text;
    }

    Analysis Analyze(const Document &doc, const Graph &graph, const Registry &registry,
                     const std::optional<std::vector<TensorType>> &inputs, int depth)
    {
        Analysis a;
        if (depth > kMaxDepth)
        {
            a.graphErrors.push_back("subgraphs nest too deeply (is a subgraph repeating itself?)");
            return a;
        }

        const Variables vars = doc.NumericVariables();

        // Resolve specs and ports, then validate links.
        std::map<int, SpecPtr> specs;
        for (const Node &n : graph.nodes)
        {
            SpecPtr spec = registry.Find(n.type);
            if (!spec) a.errors[n.id] = "unknown node type '" + n.type + "'";
            else a.ports[n.id] = spec->Ports(n, doc, registry);
            specs[n.id] = spec;
        }

        // Port order of this graph's own inputs and outputs.
        std::map<int, int> ioPosition;
        {
            const std::vector<int> ins = OrderedIo(graph, registry, NodeRole::Input);
            const std::vector<int> outs = OrderedIo(graph, registry, NodeRole::Output);
            for (size_t i = 0; i < ins.size(); ++i) ioPosition[ins[i]] = static_cast<int>(i);
            for (size_t i = 0; i < outs.size(); ++i) ioPosition[outs[i]] = static_cast<int>(i);
            a.results.resize(outs.size());
        }

        std::map<int, std::vector<const Link *>> incoming;
        std::map<int, std::set<int>> successors;
        for (const Link &l : graph.links)
        {
            if (!graph.Find(l.from.node) || !graph.Find(l.to.node))
            {
                a.graphErrors.push_back("a link refers to a missing node");
                continue;
            }
            const auto from = a.ports.find(l.from.node);
            const auto to = a.ports.find(l.to.node);
            if (from != a.ports.end() && from->second.OutputIndex(l.from.port) < 0)
            {
                a.errors[l.from.node] = "has no output named '" + l.from.port + "'";
                continue;
            }
            if (to != a.ports.end() && to->second.InputIndex(l.to.port) < 0)
            {
                a.errors[l.to.node] = "has no input named '" + l.to.port + "'";
                continue;
            }
            if (from == a.ports.end() || to == a.ports.end()) continue;
            incoming[l.to.node].push_back(&l);
            successors[l.from.node].insert(l.to.node);
        }

        // Names double as parameter prefixes, so they must be unique.
        std::map<std::string, int> names;
        for (const Node &n : graph.nodes)
        {
            if (n.name.empty()) a.errors[n.id] = "node needs a name";
            else if (names.contains(n.name)) a.errors[n.id] = "name '" + n.name + "' is used twice";
            names[n.name] = n.id;
        }

        // Kahn's algorithm; anything left over sits on a cycle.
        std::map<int, int> indegree;
        for (const Node &n : graph.nodes) indegree[n.id] = 0;
        for (const auto &[id, succ] : successors)
            for (int s : succ) ++indegree[s];
        std::queue<int> ready;
        for (const Node &n : graph.nodes) if (indegree[n.id] == 0) ready.push(n.id);
        while (!ready.empty())
        {
            const int id = ready.front();
            ready.pop();
            a.order.push_back(id);
            for (int s : successors[id]) if (--indegree[s] == 0) ready.push(s);
        }
        for (const Node &n : graph.nodes)
        {
            if (std::find(a.order.begin(), a.order.end(), n.id) == a.order.end())
            {
                a.errors[n.id] = "is part of a cycle";
            }
        }

        auto inferSubgraph = [&](const std::string &name, const std::vector<TensorType> &in) -> std::vector<TensorType>
        {
            const auto it = doc.subgraphs.find(name);
            if (it == doc.subgraphs.end()) throw GraphError("subgraph '" + name + "' does not exist");
            const Analysis sub = Analyze(doc, it->second, registry, in, depth + 1);
            if (!sub.Ok()) throw GraphError("subgraph '" + name + "' has errors: " + sub.Summary());
            return sub.results;
        };
        auto subgraphParams = [&](const std::string &name, const std::vector<TensorType> &in) -> int64_t
        {
            const auto it = doc.subgraphs.find(name);
            if (it == doc.subgraphs.end()) return 0;
            return Analyze(doc, it->second, registry, in, depth + 1).totalParams;
        };

        bool resultsComplete = true;

        for (const int id : a.order)
        {
            const Node &node = *graph.Find(id);
            const SpecPtr &spec = specs[id];
            if (!spec || a.errors.contains(id))
            {
                if (spec && spec->role == NodeRole::Output) resultsComplete = false;
                continue;
            }
            const NodePorts &ports = a.ports[id];

            std::vector<std::optional<TensorType>> in(ports.inputs.size());
            bool blocked = false;
            for (const Link *l : incoming[id])
            {
                const int port = ports.InputIndex(l->to.port);
                const auto up = a.outputs.find(l->from.node);
                if (up == a.outputs.end())
                {
                    blocked = true;   // upstream failed; its own error explains why
                    continue;
                }
                in[port] = up->second[a.ports[l->from.node].OutputIndex(l->from.port)];
            }
            if (blocked)
            {
                if (spec->role == NodeRole::Output) resultsComplete = false;
                continue;
            }

            try
            {
                for (size_t i = 0; i < ports.inputs.size(); ++i)
                {
                    const PortSpec &p = ports.inputs[i];
                    if (!in[i])
                    {
                        if (!p.optional) throw GraphError("input '" + p.name + "' is not connected");
                        continue;
                    }
                    if (p.dtype && in[i]->dtype != *p.dtype)
                    {
                        throw GraphError("input '" + p.name + "' expects " + DTypeName(*p.dtype) + " but receives " + in[i]->Text());
                    }
                }

                InferArgs args{node, graph, doc, registry, Attrs(*spec, node.attrs, vars), in, inferSubgraph, subgraphParams};

                std::vector<TensorType> out;
                if (spec->role == NodeRole::Input && inputs)
                {
                    const int index = ioPosition.at(id);
                    if (static_cast<size_t>(index) >= inputs->size())
                    {
                        throw GraphError("input '" + node.name + "' gets no value from the caller");
                    }
                    out = {(*inputs)[index]};
                }
                else
                {
                    out = spec->infer ? spec->infer(args) : std::vector<TensorType>{};
                }
                if (out.size() != ports.outputs.size())
                {
                    throw GraphError("produced " + std::to_string(out.size()) + " outputs but declares " +
                                     std::to_string(ports.outputs.size()));
                }
                for (size_t i = 0; i < out.size(); ++i)
                {
                    const PortSpec &p = ports.outputs[i];
                    if (p.dtype && out[i].dtype != *p.dtype)
                    {
                        throw GraphError("output '" + p.name + "' should be " + DTypeName(*p.dtype));
                    }
                }

                if (spec->role == NodeRole::Output) a.results[ioPosition.at(id)] = args.In(0);

                const int64_t p = spec->params ? spec->params(args) : 0;
                a.params[id] = p;
                a.totalParams += p;

                a.inputs[id] = in;
                a.outputs[id] = std::move(out);
            }
            catch (const std::exception &e)
            {
                a.errors[id] = e.what();
                if (spec->role == NodeRole::Output) resultsComplete = false;
            }
        }
        if (!resultsComplete) a.results.clear();
        return a;
    }
}
