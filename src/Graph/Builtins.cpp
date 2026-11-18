// Built-in node types. Each one is a thin adapter: the maths lives in
// InferenceEngine, the node supplies ports, attributes, shape inference and a
// parameter count so the editor can validate graphs before anything runs.

#include <cmath>
#include <map>
#include <set>
#include <sstream>

#include "Graph/Registry.h"
#include "Graph/Compiler.h"
#include "Graph/Analyze.h"

#include "Tensor/Tensor.h"
#include "NeuralNet/Linear.h"
#include "NeuralNet/Embedding.h"
#include "NeuralNet/Transformer/AttentionOps.h"
#include "NeuralNet/Transformer/GroupQueryAttention.h"
#include "NeuralNet/Transformer/MLP.h"
#include "NeuralNet/Transformer/RoPE.h"
#include "NeuralNet/Transformer/Normal/LayerNorm.h"
#include "NeuralNet/Transformer/Normal/RMSNorm.h"
#include "NeuralNet/Transformer/TransformerConfig.h"

namespace mg
{
    namespace
    {
        namespace cfgns = nn::transformer::config;
        using nn::transformer::RotaryEmbedding;

        // ---- spec helpers ----------------------------------------------------

        PortSpec F(std::string name, bool optional = false) { return {std::move(name), DType::F32, optional}; }
        PortSpec I(std::string name) { return {std::move(name), DType::I64, false}; }
        PortSpec Any(std::string name) { return {std::move(name), std::nullopt, false}; }

        AttrSpec A(std::string name, AttrKind kind, AttrValue def, std::string help, std::vector<std::string> options = {})
        {
            return {std::move(name), kind, std::move(def), std::move(options), std::move(help)};
        }

        AttrSpec IntA(std::string name, AttrValue def, std::string help) { return A(std::move(name), AttrKind::Int, std::move(def), std::move(help)); }
        AttrSpec FloatA(std::string name, AttrValue def, std::string help) { return A(std::move(name), AttrKind::Float, std::move(def), std::move(help)); }
        AttrSpec BoolA(std::string name, bool def, std::string help) { return A(std::move(name), AttrKind::Bool, def, std::move(help)); }

        const std::vector<std::string> kActivations = {"gelu", "gelu_tanh", "silu", "relu", "sigmoid", "tanh"};
        const std::vector<std::string> kRopeScaling = {"default", "linear", "llama3"};

        tensor::Activation ParseActivation(const std::string &name)
        {
            static const std::map<std::string, tensor::Activation> map = {
                {"gelu", tensor::Activation::GELU}, {"gelu_tanh", tensor::Activation::GELUTanh},
                {"silu", tensor::Activation::SiLU}, {"relu", tensor::Activation::ReLU},
                {"sigmoid", tensor::Activation::Sigmoid}, {"tanh", tensor::Activation::Tanh}};
            const auto it = map.find(name);
            if (it == map.end()) throw GraphError("unknown activation '" + name + "'");
            return it->second;
        }

        // ---- shape helpers ---------------------------------------------------

        [[noreturn]] void Fail(const std::string &message) { throw GraphError(message); }

        size_t Axis(int64_t axis, size_t rank)
        {
            const int64_t r = static_cast<int64_t>(rank);
            const int64_t a = axis < 0 ? axis + r : axis;
            if (a < 0 || a >= r) Fail("axis " + std::to_string(axis) + " is out of range for rank " + std::to_string(rank));
            return static_cast<size_t>(a);
        }

        void ExpectRank(const TensorType &t, size_t rank, const std::string &what)
        {
            if (t.Rank() != rank) Fail(ShapeError(what + " must have rank " + std::to_string(rank), t));
        }

        void ExpectDim(const TensorType &t, size_t axis, int64_t value, const std::string &what)
        {
            if (axis >= t.Rank()) Fail(ShapeError(what + " has too few dimensions", t));
            const Dim &d = t.shape[axis];
            if (d.IsKnown() && d.value != value)
            {
                Fail(ShapeError(what + " needs size " + std::to_string(value) + " at axis " + std::to_string(axis), t));
            }
        }

        void ExpectLast(const TensorType &t, int64_t value, const std::string &what)
        {
            if (t.Rank() == 0) Fail(what + " must not be a scalar");
            ExpectDim(t, t.Rank() - 1, value, what);
        }

        int64_t KnownDim(const TensorType &t, size_t axis, const std::string &what)
        {
            if (axis >= t.Rank() || !t.shape[axis].IsKnown())
            {
                Fail(ShapeError(what + " needs a known size at axis " + std::to_string(axis), t));
            }
            return t.shape[axis].value;
        }

        std::vector<Dim> Broadcast(const std::vector<Dim> &a, const std::vector<Dim> &b)
        {
            const size_t rank = std::max(a.size(), b.size());
            std::vector<Dim> out(rank);
            for (size_t i = 0; i < rank; ++i)
            {
                const Dim *da = i < rank - a.size() ? nullptr : &a[i - (rank - a.size())];
                const Dim *db = i < rank - b.size() ? nullptr : &b[i - (rank - b.size())];
                if (!da) { out[i] = *db; continue; }
                if (!db) { out[i] = *da; continue; }
                if (da->IsKnown() && da->value == 1) { out[i] = *db; continue; }
                if (db->IsKnown() && db->value == 1) { out[i] = *da; continue; }
                if (da->IsKnown() && db->IsKnown() && da->value != db->value)
                {
                    Fail("shapes do not broadcast: " + da->Text() + " vs " + db->Text());
                }
                out[i] = da->IsKnown() ? *da : *db;
            }
            return out;
        }

        TensorType WithLast(TensorType t, int64_t value)
        {
            t.shape.back() = Dim::Known(value);
            return t;
        }

        TensorType FloatType(std::vector<Dim> shape) { return TensorType{DType::F32, std::move(shape)}; }

        int64_t Product(const TensorType &t, size_t from, size_t to)
        {
            int64_t p = 1;
            for (size_t i = from; i < to; ++i) p *= KnownDim(t, i, "tensor");
            return p;
        }

        // Single-output, parameter-free node with a pure tensor function.
        template <typename Fn>
        BuildFn Pure(Fn fn)
        {
            return [fn](BuildArgs &) -> std::shared_ptr<NodeKernel>
            {
                return MakeKernel([fn](RunContext &ctx, const std::vector<Value> &in) -> std::vector<Value>
                                  { return {Value(fn(ctx, in))}; });
            };
        }

        // ---- attention helpers -----------------------------------------------

        cfgns::RopeScaling ReadScaling(const Attrs &a)
        {
            cfgns::RopeScaling s;
            const std::string type = a.Str("rope_scaling");
            if (type == "default") s.type = cfgns::RopeScalingType::Default;
            else if (type == "linear") s.type = cfgns::RopeScalingType::Linear;
            else if (type == "llama3") s.type = cfgns::RopeScalingType::Llama3;
            else Fail("unknown rope_scaling '" + type + "'");
            s.factor = a.Float("rope_factor");
            s.low_freq_factor = a.Float("rope_low_freq_factor");
            s.high_freq_factor = a.Float("rope_high_freq_factor");
            s.original_max_position = static_cast<size_t>(a.Int("rope_original_max_position"));
            return s;
        }

        std::vector<AttrSpec> RopeAttrs()
        {
            return {
                FloatA("rope_theta", 10000.0, "RoPE base frequency"),
                A("rope_scaling", AttrKind::Enum, std::string("default"), "RoPE frequency scaling", kRopeScaling),
                FloatA("rope_factor", 1.0, "scaling factor (linear, llama3)"),
                FloatA("rope_low_freq_factor", 1.0, "llama3 scaling"),
                FloatA("rope_high_freq_factor", 4.0, "llama3 scaling"),
                IntA("rope_original_max_position", int64_t{8192}, "llama3 scaling"),
                IntA("max_seq_len", std::string("$max_seq_len"), "longest sequence the RoPE table covers"),
            };
        }

        std::shared_ptr<RotaryEmbedding<Scalar>> SharedRope(BuildState &state, size_t headDim, const Attrs &a)
        {
            const cfgns::RopeScaling s = ReadScaling(a);
            const size_t maxSeq = static_cast<size_t>(a.Int("max_seq_len"));
            const double theta = a.Float("rope_theta");
            std::ostringstream key;
            key << headDim << "|" << maxSeq << "|" << theta << "|" << static_cast<int>(s.type) << "|" << s.factor << "|"
                << s.low_freq_factor << "|" << s.high_freq_factor << "|" << s.original_max_position;
            auto &slot = state.ropeCache[key.str()];
            if (!slot) slot = std::make_shared<RotaryEmbedding<Scalar>>(headDim, maxSeq, theta, s);
            return slot;
        }

        struct AttentionDims
        {
            int64_t dModel, heads, kvHeads, headDim;
        };

        AttentionDims ReadAttention(const Attrs &a)
        {
            AttentionDims d;
            d.dModel = a.Int("d_model");
            d.heads = a.Int("num_heads");
            d.kvHeads = a.Int("num_kv_heads");
            d.headDim = a.Int("head_dim");
            if (d.dModel <= 0 || d.heads <= 0) Fail("d_model and num_heads must be positive");
            if (d.kvHeads <= 0) d.kvHeads = d.heads;
            if (d.headDim <= 0)
            {
                if (d.dModel % d.heads != 0) Fail("d_model must be divisible by num_heads (or set head_dim)");
                d.headDim = d.dModel / d.heads;
            }
            if (d.heads % d.kvHeads != 0) Fail("num_heads must be divisible by num_kv_heads");
            return d;
        }

        // ---- kernels with weights --------------------------------------------

        class AttentionKernel : public NodeKernel
        {
        public:
            AttentionKernel(std::shared_ptr<nn::transformer::GroupQueryAttention<Scalar>> impl, size_t slot, bool causal)
                : m_impl(std::move(impl)), m_slot(slot), m_causal(causal) {}

            std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &in) override
            {
                auto *layer = ctx.cache ? &ctx.cache->Layer(m_slot) : nullptr;
                return {m_impl->Forward(AsFloat(in[0], "x"), layer, ctx.startPos, tensor::Tensor<bool>(), m_causal)};
            }

            std::shared_ptr<nn::Module<Scalar>> GetModule() override { return m_impl; }

        private:
            std::shared_ptr<nn::transformer::GroupQueryAttention<Scalar>> m_impl;
            size_t m_slot;
            bool m_causal;
        };

        template <typename M, typename Fn>
        class WrapKernel : public NodeKernel
        {
        public:
            WrapKernel(std::shared_ptr<M> module, Fn fn) : m_module(std::move(module)), m_fn(std::move(fn)) {}
            std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &in) override { return {Value(m_fn(*m_module, ctx, in))}; }
            std::shared_ptr<nn::Module<Scalar>> GetModule() override { return m_module; }

        private:
            std::shared_ptr<M> m_module;
            Fn m_fn;
        };

        template <typename M, typename Fn>
        std::shared_ptr<NodeKernel> Wrap(std::shared_ptr<M> module, Fn fn)
        {
            return std::make_shared<WrapKernel<M, Fn>>(std::move(module), std::move(fn));
        }

        class TiedLinearKernel : public NodeKernel
        {
        public:
            explicit TiedLinearKernel(std::string source) : m_source(std::move(source)) {}

            void Resolve(const std::function<std::shared_ptr<nn::Module<Scalar>>(const std::string &)> &find) override
            {
                auto embedding = std::dynamic_pointer_cast<nn::Embedding<Scalar>>(find(m_source));
                if (!embedding) Fail("weight_from '" + m_source + "' must name an Embedding node");
                m_weight = embedding->weight;
            }

            std::vector<Value> Run(RunContext &, const std::vector<Value> &in) override
            {
                return {tensor::Linear(AsFloat(in[0], "x"), *m_weight)};
            }

        private:
            std::string m_source;
            std::shared_ptr<FTensor> m_weight;
        };

        // Runs a subgraph once; its weights live under the Group node's name.
        class GroupKernel : public NodeKernel
        {
        public:
            explicit GroupKernel(std::shared_ptr<Program> program) : m_program(std::move(program)) {}
            std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &in) override { return m_program->Run(ctx, in); }
            std::shared_ptr<nn::Module<Scalar>> GetModule() override { return m_program; }

        private:
            std::shared_ptr<Program> m_program;
        };

        // Runs `count` copies of a subgraph. Output 0 of each copy feeds input
        // 0 of the next; the other inputs reach every copy unchanged.
        class RepeatKernel : public ModuleKernel
        {
        public:
            void Add(std::shared_ptr<Program> program)
            {
                RegisterSubmodule(std::to_string(m_programs.size()), program);
                m_programs.push_back(std::move(program));
            }

            std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &in) override
            {
                std::vector<Value> args = in;
                std::vector<Value> out;
                for (auto &p : m_programs)
                {
                    out = p->Run(ctx, args);
                    args[0] = out.at(0);
                }
                return out;
            }

        private:
            std::vector<std::shared_ptr<Program>> m_programs;
        };

        // ---- Reshape ---------------------------------------------------------

        struct ReshapeToken
        {
            enum Kind { Fixed, Copy, Infer, FromInput } kind;
            int64_t value = 0;   // Fixed: size, Copy/FromInput: input axis
            std::string symbol;
        };

        std::vector<ReshapeToken> ParseReshape(const std::string &text, const Variables &vars, const TensorType &input)
        {
            std::vector<ReshapeToken> tokens;
            std::stringstream stream(text);
            std::string t;
            int infer = 0;
            while (std::getline(stream, t, ','))
            {
                const size_t a = t.find_first_not_of(" \t");
                const size_t b = t.find_last_not_of(" \t");
                if (a == std::string::npos) Fail("reshape '" + text + "' has an empty entry");
                t = t.substr(a, b - a + 1);
                if (t == "-1")
                {
                    tokens.push_back({ReshapeToken::Infer});
                    ++infer;
                    continue;
                }
                if (t == "0")
                {
                    tokens.push_back({ReshapeToken::Copy, static_cast<int64_t>(tokens.size())});
                    continue;
                }
                if (IsIdentifier(t) && !vars.contains(t))
                {
                    bool found = false;
                    for (size_t i = 0; i < input.Rank(); ++i)
                    {
                        if (input.shape[i].symbol == t)
                        {
                            tokens.push_back({ReshapeToken::FromInput, static_cast<int64_t>(i), t});
                            found = true;
                            break;
                        }
                    }
                    if (!found) Fail("reshape: symbol '" + t + "' does not appear in the input shape " + input.Text());
                    continue;
                }
                const double v = Evaluate(t, vars);
                if (v <= 0) Fail("reshape sizes must be positive");
                tokens.push_back({ReshapeToken::Fixed, static_cast<int64_t>(std::llround(v))});
            }
            if (infer > 1) Fail("reshape may use -1 only once");
            return tokens;
        }

        // Symbolic element count: known factor times a multiset of symbols.
        struct Count
        {
            int64_t known = 1;
            std::multiset<std::string> symbols;
            bool unknown = false;
            void Mul(const Dim &d)
            {
                if (d.IsKnown()) known *= d.value;
                else if (!d.symbol.empty() && d.symbol != "?") symbols.insert(d.symbol);
                else unknown = true;
            }
        };

        TensorType InferReshape(const TensorType &in, const std::vector<ReshapeToken> &tokens)
        {
            TensorType out{in.dtype, {}};
            Count total, rest;
            for (const Dim &d : in.shape) total.Mul(d);
            int inferAt = -1;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                const ReshapeToken &t = tokens[i];
                Dim d;
                switch (t.kind)
                {
                case ReshapeToken::Fixed: d = Dim::Known(t.value); break;
                case ReshapeToken::Copy:
                    if (static_cast<size_t>(t.value) >= in.Rank()) Fail("reshape: 0 at a position the input does not have");
                    d = in.shape[t.value];
                    break;
                case ReshapeToken::FromInput: d = in.shape[t.value]; break;
                case ReshapeToken::Infer: inferAt = static_cast<int>(i); break;
                }
                if (t.kind != ReshapeToken::Infer) rest.Mul(d);
                out.shape.push_back(d);
            }
            if (inferAt >= 0)
            {
                Dim d = Dim::Sym("?");
                bool ok = !total.unknown && !rest.unknown && rest.known != 0 && total.known % rest.known == 0;
                std::multiset<std::string> left = total.symbols;
                for (const std::string &s : rest.symbols)
                {
                    const auto it = left.find(s);
                    if (it == left.end()) { ok = false; break; }
                    left.erase(it);
                }
                if (ok)
                {
                    const int64_t q = total.known / rest.known;
                    if (left.empty()) d = Dim::Known(q);
                    else if (left.size() == 1 && q == 1) d = Dim::Sym(*left.begin());
                }
                out.shape[inferAt] = d;
            }
            else if (!total.unknown && !rest.unknown && total.symbols == rest.symbols && total.known != rest.known)
            {
                Fail("reshape from " + in.Text() + " to " + out.Text() + " changes the element count");
            }
            return out;
        }

        // ---- the node table --------------------------------------------------

        void RegisterIO(Registry &r)
        {
            {
                NodeSpec s;
                s.type = "TokenIds";
                s.category = "Inputs & Outputs";
                s.description = "The model input: token ids of shape [B, S].";
                s.outputs = {I("ids")};
                s.role = NodeRole::Input;
                s.infer = [](const InferArgs &) { return std::vector<TensorType>{{DType::I64, {Dim::Sym("B"), Dim::Sym("S")}}}; };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Logits";
                s.category = "Inputs & Outputs";
                s.description = "The model output: next-token scores of shape [B, S, vocab].";
                s.inputs = {F("logits")};
                s.role = NodeRole::Output;
                s.infer = [](const InferArgs &a)
                {
                    if (a.In(0).Rank() != 3) Fail(ShapeError("logits must be [B, S, vocab]", a.In(0)));
                    return std::vector<TensorType>{};
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Input";
                s.category = "Inputs & Outputs";
                s.description = "A subgraph input. When the subgraph is used by Repeat, the caller's tensor replaces the declared shape.";
                s.outputs = {Any("out")};
                s.role = NodeRole::Input;
                s.attrs = {IntA("index", int64_t{0}, "position among the subgraph inputs"),
                           A("dtype", AttrKind::Enum, std::string("f32"), "element type", {"f32", "i64"}),
                           A("shape", AttrKind::Shape, std::string("B, S, $d_model"), "declared shape (names like B, S are symbolic)")};
                s.infer = [](const InferArgs &a)
                {
                    return std::vector<TensorType>{{ParseDType(a.attrs.Str("dtype")), ParseShape(a.attrs.Str("shape"), a.attrs.Vars())}};
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Output";
                s.category = "Inputs & Outputs";
                s.description = "A subgraph output.";
                s.inputs = {Any("value")};
                s.role = NodeRole::Output;
                s.attrs = {IntA("index", int64_t{0}, "position among the subgraph outputs")};
                s.infer = [](const InferArgs &a)
                {
                    (void)a.In(0);
                    return std::vector<TensorType>{};
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Positions";
                s.category = "Inputs & Outputs";
                s.description = "Absolute positions [S] of the tokens in this call (offset by what the KV cache already holds).";
                s.inputs = {I("ids")};
                s.outputs = {I("positions")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectRank(a.In(0), 2, "ids");
                    return std::vector<TensorType>{{DType::I64, {a.In(0).shape[1]}}};
                };
                s.build = Pure([](RunContext &ctx, const std::vector<Value> &in) -> Value
                {
                    const size_t seq = AsInt(in[0], "ids").Shape()[1];
                    ITensor pos({seq});
                    for (size_t i = 0; i < seq; ++i) pos(i) = static_cast<int64_t>(ctx.startPos + i);
                    return pos.To(ctx.device);
                });
                r.Register(std::move(s));
            }
        }

        void RegisterLayers(Registry &r)
        {
            {
                NodeSpec s;
                s.type = "Embedding";
                s.category = "Layers";
                s.description = "Looks up a learned vector for every token id.";
                s.inputs = {I("ids")};
                s.outputs = {F("out")};
                s.attrs = {IntA("num_embeddings", std::string("$vocab_size"), "table rows (vocabulary size)"),
                           IntA("dim", std::string("$d_model"), "vector size")};
                s.infer = [](const InferArgs &a)
                {
                    TensorType t = a.In(0);
                    t.dtype = DType::F32;
                    t.shape.push_back(Dim::Known(a.attrs.Int("dim")));
                    return std::vector<TensorType>{t};
                };
                s.params = [](const InferArgs &a) { return a.attrs.Int("num_embeddings") * a.attrs.Int("dim"); };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    auto m = std::make_shared<nn::Embedding<Scalar>>(static_cast<size_t>(b.attrs.Int("num_embeddings")),
                                                                     static_cast<size_t>(b.attrs.Int("dim")));
                    return Wrap(m, [](nn::Embedding<Scalar> &e, RunContext &, const std::vector<Value> &in) -> Value
                                { return e.Forward(AsInt(in[0], "ids")); });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Linear";
                s.category = "Layers";
                s.description = "y = x W^T + b";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("in_features", std::string("$d_model"), "input size"),
                           IntA("out_features", std::string("$d_model"), "output size"),
                           BoolA("bias", true, "add a learned bias")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectLast(a.In(0), a.attrs.Int("in_features"), "x");
                    return std::vector<TensorType>{WithLast(a.In(0), a.attrs.Int("out_features"))};
                };
                s.params = [](const InferArgs &a)
                {
                    const int64_t in = a.attrs.Int("in_features"), out = a.attrs.Int("out_features");
                    return in * out + (a.attrs.Bool("bias") ? out : 0);
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    auto m = std::make_shared<nn::Linear<Scalar>>(static_cast<size_t>(b.attrs.Int("in_features")),
                                                                  static_cast<size_t>(b.attrs.Int("out_features")), b.attrs.Bool("bias"));
                    return Wrap(m, [](nn::Linear<Scalar> &l, RunContext &, const std::vector<Value> &in) -> Value
                                { return l.Forward(AsFloat(in[0], "x")); });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "TiedLinear";
                s.category = "Layers";
                s.description = "Output projection that reuses an Embedding's table as its weight (weight tying).";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {A("weight_from", AttrKind::NodeRef, std::string("wte"), "name of the Embedding node to share")};
                s.infer = [](const InferArgs &a)
                {
                    const std::string source = a.attrs.Str("weight_from");
                    const Node *n = a.graph.FindByName(source);
                    if (!n || n->type != "Embedding") Fail("weight_from '" + source + "' must name an Embedding node in this graph");
                    const SpecPtr spec = a.registry.Find("Embedding");
                    const Attrs ea(*spec, n->attrs, a.attrs.Vars());
                    ExpectLast(a.In(0), ea.Int("dim"), "x");
                    return std::vector<TensorType>{WithLast(a.In(0), ea.Int("num_embeddings"))};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                { return std::make_shared<TiedLinearKernel>(b.attrs.Str("weight_from")); };
                r.Register(std::move(s));
            }

            for (const bool rms : {false, true})
            {
                NodeSpec s;
                s.type = rms ? "RMSNorm" : "LayerNorm";
                s.category = "Layers";
                s.description = rms ? "x / rms(x) * weight" : "(x - mean) / std * gamma + beta";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("dim", std::string("$d_model"), "normalised (last) dimension"),
                           FloatA("eps", 1e-5, "added to the variance")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectLast(a.In(0), a.attrs.Int("dim"), "x");
                    return std::vector<TensorType>{a.In(0)};
                };
                s.params = [rms](const InferArgs &a) { return a.attrs.Int("dim") * (rms ? 1 : 2); };
                s.build = [rms](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const size_t dim = static_cast<size_t>(b.attrs.Int("dim"));
                    const Scalar eps = static_cast<Scalar>(b.attrs.Float("eps"));
                    auto run = [](auto &m, RunContext &, const std::vector<Value> &in) -> Value { return m.Forward(AsFloat(in[0], "x")); };
                    if (rms) return Wrap(std::make_shared<nn::transformer::norm::RMSNorm<Scalar>>(dim, eps), run);
                    return Wrap(std::make_shared<nn::transformer::norm::LayerNorm<Scalar>>(dim, eps), run);
                };
                r.Register(std::move(s));
            }
        }

        std::vector<TensorType> AllInputs(const InferArgs &a)
        {
            std::vector<TensorType> in;
            for (size_t i = 0; i < a.inputs.size(); ++i) in.push_back(a.In(i));
            return in;
        }

        std::vector<TensorType> AllInputs(const BuildArgs &b)
        {
            std::vector<TensorType> in;
            for (size_t i = 0; i < b.inputs.size(); ++i) in.push_back(b.In(i));
            return in;
        }

        NodePorts NestPorts(const Node &node, const Document &doc, const Registry &registry)
        {
            const auto it = node.attrs.find("subgraph");
            return it == node.attrs.end() ? NodePorts{} : SubgraphPorts(doc, AttrText(it->second), registry);
        }

        void RegisterNests(Registry &r)
        {
            {
                NodeSpec s;
                s.type = "Group";
                s.category = "Structure";
                s.description = "A nest: a box of nodes defined by a subgraph. Its ports are the subgraph's Input and Output nodes, "
                                "and its weights are named after this node (e.g. mlp.up.weight).";
                s.attrs = {A("subgraph", AttrKind::Subgraph, std::string(""), "the subgraph this box contains")};
                s.dynamicPorts = NestPorts;
                s.infer = [](const InferArgs &a)
                {
                    const std::string sub = a.attrs.Str("subgraph");
                    if (sub.empty()) Fail("choose a subgraph");
                    return a.inferSubgraph(sub, AllInputs(a));
                };
                s.params = [](const InferArgs &a) { return a.subgraphParams(a.attrs.Str("subgraph"), AllInputs(a)); };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                { return std::make_shared<GroupKernel>(b.state.buildSubgraph(b.attrs.Str("subgraph"), AllInputs(b))); };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Repeat";
                s.category = "Structure";
                s.description = "Runs a subgraph `count` times, each copy with its own weights (name.0, name.1, ...). "
                                "Output 0 of each copy feeds input 0 of the next; other inputs reach every copy.";
                s.attrs = {A("subgraph", AttrKind::Subgraph, std::string(""), "the subgraph to repeat"),
                           IntA("count", std::string("$num_layers"), "number of copies")};
                s.dynamicPorts = NestPorts;
                s.infer = [](const InferArgs &a)
                {
                    const std::string sub = a.attrs.Str("subgraph");
                    if (sub.empty()) Fail("choose a subgraph");
                    const int64_t count = a.attrs.Int("count");
                    if (count < 1) Fail("count must be at least 1");
                    const std::vector<TensorType> in = AllInputs(a);
                    if (in.empty()) Fail("the repeated subgraph needs at least one Input");
                    const std::vector<TensorType> out = a.inferSubgraph(sub, in);
                    if (out.empty()) Fail("the repeated subgraph needs at least one Output");
                    if (count > 1 && (out[0].shape != in[0].shape || out[0].dtype != in[0].dtype))
                    {
                        Fail("output 0 feeds input 0 of the next copy, so it must keep the shape: " + in[0].Text() + " -> " + out[0].Text());
                    }
                    return out;
                };
                s.params = [](const InferArgs &a) { return a.attrs.Int("count") * a.subgraphParams(a.attrs.Str("subgraph"), AllInputs(a)); };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    auto kernel = std::make_shared<RepeatKernel>();
                    const int64_t count = b.attrs.Int("count");
                    const std::vector<TensorType> in = AllInputs(b);
                    for (int64_t i = 0; i < count; ++i) kernel->Add(b.state.buildSubgraph(b.attrs.Str("subgraph"), in));
                    return kernel;
                };
                r.Register(std::move(s));
            }
        }

        void RegisterTransformer(Registry &r)
        {
            {
                NodeSpec s;
                s.type = "MLP";
                s.category = "Transformer";
                s.description = "Fused feed-forward block: down(act(up(x))), or down(act(gate(x)) * up(x)) when gated. "
                                "For a customisable version, build it from Linear and Activation nodes inside a Group.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("d_model", std::string("$d_model"), "model width"),
                           IntA("ffn_dim", std::string("$ffn_dim"), "hidden width"),
                           BoolA("gated", false, "gated (SwiGLU-style) variant"),
                           A("activation", AttrKind::Enum, std::string("gelu"), "activation function", kActivations),
                           BoolA("bias", true, "biases on the projections")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectLast(a.In(0), a.attrs.Int("d_model"), "x");
                    return std::vector<TensorType>{a.In(0)};
                };
                s.params = [](const InferArgs &a)
                {
                    const int64_t d = a.attrs.Int("d_model"), f = a.attrs.Int("ffn_dim");
                    const bool gated = a.attrs.Bool("gated"), bias = a.attrs.Bool("bias");
                    return (gated ? 3 : 2) * d * f + (bias ? (gated ? 2 * f : f) + d : 0);
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    cfgns::TransformerConfig cfg;
                    cfg.d_model = static_cast<size_t>(b.attrs.Int("d_model"));
                    cfg.ffn_dim = static_cast<size_t>(b.attrs.Int("ffn_dim"));
                    cfg.mlp_type = b.attrs.Bool("gated") ? cfgns::MLPType::Gated : cfgns::MLPType::Standard;
                    cfg.activation = ParseActivation(b.attrs.Str("activation"));
                    cfg.output_bias = b.attrs.Bool("bias");
                    auto m = std::make_shared<nn::transformer::MLP<Scalar>>(cfg);
                    return Wrap(m, [](nn::transformer::MLP<Scalar> &mlp, RunContext &, const std::vector<Value> &in) -> Value
                                { return mlp.Forward(AsFloat(in[0], "x")); });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Attention";
                s.category = "Transformer";
                s.description = "Fused causal self-attention (grouped-query heads, optional RoPE, KV cache). "
                                "For a customisable version, build it from Linear, SplitHeads, RoPE, KVCache and SDPA inside a Group.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("d_model", std::string("$d_model"), "model width"),
                           IntA("num_heads", std::string("$num_heads"), "query heads"),
                           IntA("num_kv_heads", int64_t{0}, "key/value heads (0 = same as num_heads)"),
                           IntA("head_dim", int64_t{0}, "per-head size (0 = d_model / num_heads)"),
                           BoolA("qkv_bias", true, "bias on q, k, v projections"),
                           BoolA("out_bias", true, "bias on the output projection"),
                           BoolA("causal", true, "mask future tokens"),
                           BoolA("rope", false, "apply rotary position embeddings")};
                for (AttrSpec &a : RopeAttrs()) s.attrs.push_back(std::move(a));
                s.infer = [](const InferArgs &a)
                {
                    const AttentionDims d = ReadAttention(a.attrs);
                    ExpectRank(a.In(0), 3, "x");
                    ExpectLast(a.In(0), d.dModel, "x");
                    if (a.attrs.Bool("rope") && d.headDim % 2 != 0) Fail("RoPE needs an even head_dim");
                    return std::vector<TensorType>{a.In(0)};
                };
                s.params = [](const InferArgs &a)
                {
                    const AttentionDims d = ReadAttention(a.attrs);
                    const int64_t q = d.heads * d.headDim, kv = d.kvHeads * d.headDim;
                    int64_t p = d.dModel * (q + 2 * kv) + q * d.dModel;
                    if (a.attrs.Bool("qkv_bias")) p += q + 2 * kv;
                    if (a.attrs.Bool("out_bias")) p += d.dModel;
                    return p;
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const AttentionDims d = ReadAttention(b.attrs);
                    cfgns::TransformerConfig cfg;
                    cfg.d_model = static_cast<size_t>(d.dModel);
                    cfg.num_heads = static_cast<size_t>(d.heads);
                    cfg.num_kv_heads = static_cast<size_t>(d.kvHeads);
                    cfg.head_dim = static_cast<size_t>(d.headDim);
                    cfg.attention_bias = b.attrs.Bool("qkv_bias");
                    cfg.output_bias = b.attrs.Bool("out_bias");
                    const bool rope = b.attrs.Bool("rope");
                    cfg.position = rope ? cfgns::PositionEncoding::Rotary : cfgns::PositionEncoding::None;
                    std::shared_ptr<RotaryEmbedding<Scalar>> table = rope ? SharedRope(b.state, cfg.head_dim, b.attrs) : nullptr;
                    auto impl = std::make_shared<nn::transformer::GroupQueryAttention<Scalar>>(cfg, table);
                    const size_t slot = b.state.AllocateSlot({cfg.num_kv_heads, cfg.head_dim});
                    return std::make_shared<AttentionKernel>(impl, slot, b.attrs.Bool("causal"));
                };
                r.Register(std::move(s));
            }
            RegisterNests(r);
        }

        void RegisterMath(Registry &r)
        {
            struct Binary
            {
                const char *type;
                const char *desc;
                FTensor (*fn)(const FTensor &, const FTensor &);
            };
            const Binary binaries[] = {
                {"Add", "a + b (broadcasting)", [](const FTensor &a, const FTensor &b) { return a + b; }},
                {"Sub", "a - b (broadcasting)", [](const FTensor &a, const FTensor &b) { return a - b; }},
                {"Mul", "a * b element-wise (broadcasting)", [](const FTensor &a, const FTensor &b) { return a * b; }},
                {"Div", "a / b element-wise (broadcasting)", [](const FTensor &a, const FTensor &b) { return a / b; }},
            };
            for (const Binary &op : binaries)
            {
                NodeSpec s;
                s.type = op.type;
                s.category = "Math";
                s.description = op.desc;
                s.inputs = {F("a"), F("b")};
                s.outputs = {F("out")};
                s.infer = [](const InferArgs &a) { return std::vector<TensorType>{FloatType(Broadcast(a.In(0).shape, a.In(1).shape))}; };
                auto fn = op.fn;
                s.build = Pure([fn](RunContext &, const std::vector<Value> &in) -> Value
                               { return fn(AsFloat(in[0], "a"), AsFloat(in[1], "b")); });
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Scale";
                s.category = "Math";
                s.description = "x * factor + offset";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {FloatA("factor", 1.0, "multiplier"), FloatA("offset", 0.0, "added afterwards")};
                s.infer = [](const InferArgs &a) { return std::vector<TensorType>{a.In(0)}; };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const Scalar f = static_cast<Scalar>(b.attrs.Float("factor"));
                    const Scalar o = static_cast<Scalar>(b.attrs.Float("offset"));
                    return MakeKernel([f, o](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                    {
                        FTensor y = AsFloat(in[0], "x") * f;
                        if (o != 0) y += o;
                        return {y};
                    });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Activation";
                s.category = "Math";
                s.description = "Element-wise non-linearity.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {A("kind", AttrKind::Enum, std::string("gelu"), "function", kActivations)};
                s.infer = [](const InferArgs &a)
                {
                    ParseActivation(a.attrs.Str("kind"));
                    return std::vector<TensorType>{a.In(0)};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const tensor::Activation kind = ParseActivation(b.attrs.Str("kind"));
                    return MakeKernel([kind](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                                      { return {tensor::Activate(AsFloat(in[0], "x"), kind)}; });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Softmax";
                s.category = "Math";
                s.description = "Softmax along an axis.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("axis", int64_t{-1}, "axis to normalise")};
                s.infer = [](const InferArgs &a)
                {
                    Axis(a.attrs.Int("axis"), a.In(0).Rank());
                    return std::vector<TensorType>{a.In(0)};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const long long axis = b.attrs.Int("axis");
                    return MakeKernel([axis](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                                      { return {tensor::Softmax(AsFloat(in[0], "x"), axis)}; });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Mean";
                s.category = "Math";
                s.description = "Mean along an axis.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("axis", int64_t{-1}, "axis to reduce"), BoolA("keepdim", true, "keep the reduced axis as size 1")};
                s.infer = [](const InferArgs &a)
                {
                    TensorType t = a.In(0);
                    const size_t axis = Axis(a.attrs.Int("axis"), t.Rank());
                    if (a.attrs.Bool("keepdim")) t.shape[axis] = Dim::Known(1);
                    else t.shape.erase(t.shape.begin() + static_cast<long>(axis));
                    return std::vector<TensorType>{t};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const int64_t axis = b.attrs.Int("axis");
                    const bool keep = b.attrs.Bool("keepdim");
                    return MakeKernel([axis, keep](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                    {
                        const FTensor &x = AsFloat(in[0], "x");
                        return {tensor::Mean(x, Axis(axis, x.Rank()), keep)};
                    });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Sqrt";
                s.category = "Math";
                s.description = "Element-wise square root.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.infer = [](const InferArgs &a) { return std::vector<TensorType>{a.In(0)}; };
                s.build = Pure([](RunContext &, const std::vector<Value> &in) -> Value { return tensor::Sqrt(AsFloat(in[0], "x")); });
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "MatMul";
                s.category = "Math";
                s.description = "Batched matrix product a @ b over the last two axes.";
                s.inputs = {F("a"), F("b")};
                s.outputs = {F("out")};
                s.infer = [](const InferArgs &a)
                {
                    const TensorType &x = a.In(0), &y = a.In(1);
                    if (x.Rank() < 2 || y.Rank() < 2) Fail("MatMul needs rank >= 2 inputs");
                    const Dim &k1 = x.shape[x.Rank() - 1], &k2 = y.shape[y.Rank() - 2];
                    if (k1.IsKnown() && k2.IsKnown() && k1.value != k2.value)
                    {
                        Fail("inner sizes differ: " + x.Text() + " @ " + y.Text());
                    }
                    std::vector<Dim> bx(x.shape.begin(), x.shape.end() - 2), by(y.shape.begin(), y.shape.end() - 2);
                    std::vector<Dim> out = Broadcast(bx, by);
                    out.push_back(x.shape[x.Rank() - 2]);
                    out.push_back(y.shape[y.Rank() - 1]);
                    return std::vector<TensorType>{FloatType(out)};
                };
                s.build = Pure([](RunContext &, const std::vector<Value> &in) -> Value
                               { return tensor::MatMul(AsFloat(in[0], "a"), AsFloat(in[1], "b")); });
                r.Register(std::move(s));
            }
        }

        void RegisterTensorOps(Registry &r)
        {
            {
                NodeSpec s;
                s.type = "Transpose";
                s.category = "Tensor";
                s.description = "Swaps two axes.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {IntA("dim0", int64_t{-2}, "first axis"), IntA("dim1", int64_t{-1}, "second axis")};
                s.infer = [](const InferArgs &a)
                {
                    TensorType t = a.In(0);
                    std::swap(t.shape[Axis(a.attrs.Int("dim0"), t.Rank())], t.shape[Axis(a.attrs.Int("dim1"), t.Rank())]);
                    return std::vector<TensorType>{t};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const long long d0 = b.attrs.Int("dim0"), d1 = b.attrs.Int("dim1");
                    return MakeKernel([d0, d1](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                                      { return {AsFloat(in[0], "x").Transpose(d0, d1).Contiguous()}; });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "Reshape";
                s.category = "Tensor";
                s.description = "Reshape. Entries: sizes or $expressions, 0 = keep that axis, -1 = infer, or a symbol from the input (B, S).";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = {A("shape", AttrKind::Shape, std::string("B, S, -1"), "target shape")};
                s.infer = [](const InferArgs &a)
                {
                    return std::vector<TensorType>{InferReshape(a.In(0), ParseReshape(a.attrs.Str("shape"), a.attrs.Vars(), a.In(0)))};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const std::vector<ReshapeToken> tokens = ParseReshape(b.attrs.Str("shape"), b.attrs.Vars(), b.In(0));
                    return MakeKernel([tokens](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                    {
                        const FTensor &x = AsFloat(in[0], "x");
                        std::vector<size_t> shape;
                        size_t known = 1;
                        int inferAt = -1;
                        for (const ReshapeToken &t : tokens)
                        {
                            size_t d = 0;
                            switch (t.kind)
                            {
                            case ReshapeToken::Fixed: d = static_cast<size_t>(t.value); break;
                            case ReshapeToken::Copy:
                            case ReshapeToken::FromInput: d = x.Shape().at(static_cast<size_t>(t.value)); break;
                            case ReshapeToken::Infer: inferAt = static_cast<int>(shape.size()); d = 1; break;
                            }
                            if (t.kind != ReshapeToken::Infer) known *= d;
                            shape.push_back(d);
                        }
                        if (inferAt >= 0)
                        {
                            if (known == 0 || x.Size() % known != 0) Fail("cannot infer the -1 dimension");
                            shape[inferAt] = x.Size() / known;
                        }
                        return {x.Contiguous().Reshape(shape)};
                    });
                };
                r.Register(std::move(s));
            }
        }

        void RegisterAttentionParts(Registry &r)
        {
            const std::string cat = "Attention parts";
            {
                NodeSpec s;
                s.type = "SplitHeads";
                s.category = cat;
                s.description = "[B, S, H*D] -> [B, H, S, D]";
                s.inputs = {F("x")};
                s.outputs = {F("heads")};
                s.attrs = {IntA("num_heads", std::string("$num_heads"), "H"), IntA("head_dim", std::string("$d_model / $num_heads"), "D")};
                s.infer = [](const InferArgs &a)
                {
                    const int64_t h = a.attrs.Int("num_heads"), d = a.attrs.Int("head_dim");
                    ExpectRank(a.In(0), 3, "x");
                    ExpectLast(a.In(0), h * d, "x");
                    const auto &in = a.In(0).shape;
                    return std::vector<TensorType>{FloatType({in[0], Dim::Known(h), in[1], Dim::Known(d)})};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const size_t h = static_cast<size_t>(b.attrs.Int("num_heads")), d = static_cast<size_t>(b.attrs.Int("head_dim"));
                    return MakeKernel([h, d](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                                      { return {nn::transformer::SplitHeads(AsFloat(in[0], "x").Contiguous(), h, d)}; });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "MergeHeads";
                s.category = cat;
                s.description = "[B, H, S, D] -> [B, S, H*D]";
                s.inputs = {F("heads")};
                s.outputs = {F("x")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectRank(a.In(0), 4, "heads");
                    const auto &in = a.In(0).shape;
                    const int64_t width = KnownDim(a.In(0), 1, "heads") * KnownDim(a.In(0), 3, "heads");
                    return std::vector<TensorType>{FloatType({in[0], in[2], Dim::Known(width)})};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const size_t width = static_cast<size_t>(Product(b.In(0), 1, 2) * Product(b.In(0), 3, 4));
                    return MakeKernel([width](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                                      { return {nn::transformer::MergeHeads(AsFloat(in[0], "heads").Contiguous(), width)}; });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "RoPE";
                s.category = cat;
                s.description = "Rotary position embedding on [B, H, S, D], offset by the cache position.";
                s.inputs = {F("x")};
                s.outputs = {F("y")};
                s.attrs = RopeAttrs();
                s.infer = [](const InferArgs &a)
                {
                    ExpectRank(a.In(0), 4, "x");
                    if (KnownDim(a.In(0), 3, "x") % 2 != 0) Fail("RoPE needs an even head_dim");
                    ReadScaling(a.attrs);
                    return std::vector<TensorType>{a.In(0)};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    auto table = SharedRope(b.state, static_cast<size_t>(KnownDim(b.In(0), 3, "x")), b.attrs);
                    return Wrap(table, [](RotaryEmbedding<Scalar> &rope, RunContext &ctx, const std::vector<Value> &in) -> Value
                                { return rope.Apply(AsFloat(in[0], "x"), ctx.startPos); });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "KVCache";
                s.category = cat;
                s.description = "Appends this step's keys/values [B, Hkv, S, D] to the cache and returns everything so far [B, Hkv, T, D].";
                s.inputs = {F("k"), F("v")};
                s.outputs = {F("k_all"), F("v_all")};
                s.infer = [](const InferArgs &a)
                {
                    ExpectRank(a.In(0), 4, "k");
                    ExpectRank(a.In(1), 4, "v");
                    KnownDim(a.In(0), 1, "k");
                    KnownDim(a.In(0), 3, "k");
                    if (a.In(0).shape != a.In(1).shape) Fail("k and v must have the same shape");
                    TensorType t = a.In(0);
                    t.shape[2] = Dim::Sym("T");
                    return std::vector<TensorType>{t, t};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const size_t slot = b.state.AllocateSlot({static_cast<size_t>(KnownDim(b.In(0), 1, "k")),
                                                              static_cast<size_t>(KnownDim(b.In(0), 3, "k"))});
                    return MakeKernel([slot](RunContext &ctx, const std::vector<Value> &in) -> std::vector<Value>
                    {
                        const FTensor &k = AsFloat(in[0], "k");
                        const FTensor &v = AsFloat(in[1], "v");
                        if (!ctx.cache) return {k, v};
                        auto &layer = ctx.cache->Layer(slot);
                        const size_t seq = k.Shape()[2];
                        tensor::CopyInto(k, layer.k.Slice(2, ctx.startPos, seq));
                        tensor::CopyInto(v, layer.v.Slice(2, ctx.startPos, seq));
                        return {layer.k.Slice(2, 0, ctx.startPos + seq), layer.v.Slice(2, 0, ctx.startPos + seq)};
                    });
                };
                r.Register(std::move(s));
            }
            {
                NodeSpec s;
                s.type = "SDPA";
                s.category = cat;
                s.description = "softmax(q k^T * scale) v on [B, H, S, D]; k/v may have fewer heads (grouped-query).";
                s.inputs = {F("q"), F("k"), F("v")};
                s.outputs = {F("out")};
                s.attrs = {BoolA("causal", true, "mask future tokens"), FloatA("scale", 0.0, "0 = 1/sqrt(head_dim)")};
                s.infer = [](const InferArgs &a)
                {
                    for (size_t i = 0; i < 3; ++i) ExpectRank(a.In(i), 4, "q/k/v");
                    const int64_t hq = KnownDim(a.In(0), 1, "q"), hk = KnownDim(a.In(1), 1, "k");
                    if (hq % hk != 0) Fail("query heads must be a multiple of key/value heads");
                    ExpectDim(a.In(1), 3, KnownDim(a.In(0), 3, "q"), "k");
                    return std::vector<TensorType>{a.In(0)};
                };
                s.build = [](BuildArgs &b) -> std::shared_ptr<NodeKernel>
                {
                    const bool causal = b.attrs.Bool("causal");
                    const double scaleAttr = b.attrs.Float("scale");
                    return MakeKernel([causal, scaleAttr](RunContext &, const std::vector<Value> &in) -> std::vector<Value>
                    {
                        const FTensor q = AsFloat(in[0], "q").Contiguous();
                        const FTensor &k = AsFloat(in[1], "k");
                        const FTensor &v = AsFloat(in[2], "v");
                        const auto &qs = q.Shape();
                        const size_t hk = k.Shape()[1];
                        const Scalar scale = static_cast<Scalar>(scaleAttr != 0 ? scaleAttr : 1.0 / std::sqrt(static_cast<double>(qs[3])));
                        if (qs[1] == hk) return {nn::transformer::SDPA(q, k, v, tensor::Tensor<bool>(), causal, scale)};
                        const size_t groups = qs[1] / hk;
                        const FTensor q5 = q.Reshape(qs[0], hk, groups, qs[2], qs[3]);
                        const FTensor out = nn::transformer::SDPA(q5, k.Unsqueeze(2), v.Unsqueeze(2), tensor::Tensor<bool>(), causal, scale);
                        return {out.Contiguous().Reshape(qs[0], qs[1], qs[2], qs[3])};
                    });
                };
                r.Register(std::move(s));
            }
        }
    }

    void RegisterBuiltins(Registry &registry)
    {
        RegisterIO(registry);
        RegisterLayers(registry);
        RegisterTransformer(registry);
        RegisterMath(registry);
        RegisterTensorOps(registry);
        RegisterAttentionParts(registry);
    }
}
