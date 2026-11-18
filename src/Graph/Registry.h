#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "Graph/Document.h"
#include "Graph/Runtime.h"

namespace nn::transformer { template <typename T> class RotaryEmbedding; }

namespace mg
{
    enum class AttrKind { Bool, Int, Float, String, Enum, Shape, Subgraph, NodeRef };

    struct AttrSpec
    {
        std::string name;
        AttrKind kind = AttrKind::Int;
        AttrValue def = int64_t{0};
        std::vector<std::string> options;   // Enum choices
        std::string help;
    };

    struct PortSpec
    {
        std::string name;
        std::optional<DType> dtype = DType::F32;   // nullopt: any dtype
        bool optional = false;
    };

    enum class NodeRole { Op, Input, Output };

    // The ports of one node. Usually fixed by the node type; Group and Repeat
    // nodes take theirs from the Input/Output nodes of their subgraph.
    struct NodePorts
    {
        std::vector<PortSpec> inputs;
        std::vector<PortSpec> outputs;

        int InputIndex(const std::string &name) const;
        int OutputIndex(const std::string &name) const;
    };

    struct NodeSpec;
    class Registry;

    // Typed, expression-aware view over a node's attributes.
    class Attrs
    {
    public:
        // Holds the variables by value: callers often pass a temporary
        // (doc.NumericVariables()), and the map is small.
        Attrs(const NodeSpec &spec, const AttrMap &values, Variables vars)
            : m_spec(spec), m_values(values), m_vars(std::move(vars)) {}

        int64_t Int(const std::string &name) const;
        double Float(const std::string &name) const;
        bool Bool(const std::string &name) const;
        std::string Str(const std::string &name) const;
        AttrValue Raw(const std::string &name) const;
        const Variables &Vars() const { return m_vars; }

    private:
        const NodeSpec &m_spec;
        const AttrMap &m_values;
        Variables m_vars;
    };

    class Registry;

    struct InferArgs
    {
        const Node &node;
        const Graph &graph;
        const Document &doc;
        const Registry &registry;
        Attrs attrs;
        std::vector<std::optional<TensorType>> inputs;
        // Analyses a subgraph given its input types; returns its output types.
        std::function<std::vector<TensorType>(const std::string &, const std::vector<TensorType> &)> inferSubgraph;
        // Parameter count of a subgraph called with the given input types.
        std::function<int64_t(const std::string &, const std::vector<TensorType> &)> subgraphParams;

        const TensorType &In(size_t i) const;
    };

    class Program;

    struct BuildState
    {
        const Document &doc;
        const Registry &registry;
        std::vector<KVSlotSpec> slots;
        std::map<std::string, std::shared_ptr<nn::transformer::RotaryEmbedding<Scalar>>> ropeCache;
        std::function<std::shared_ptr<Program>(const std::string &, const std::vector<TensorType> &)> buildSubgraph;

        size_t AllocateSlot(KVSlotSpec spec)
        {
            slots.push_back(spec);
            return slots.size() - 1;
        }
    };

    struct BuildArgs
    {
        const Node &node;
        Attrs attrs;
        std::vector<std::optional<TensorType>> inputs;   // inferred input types, by port
        std::vector<TensorType> outputs;                 // inferred output types
        BuildState &state;

        const TensorType &In(size_t i) const
        {
            if (i >= inputs.size() || !inputs[i]) throw GraphError("input " + std::to_string(i) + " is not connected");
            return *inputs[i];
        }
    };

    using InferFn = std::function<std::vector<TensorType>(const InferArgs &)>;
    using ParamsFn = std::function<int64_t(const InferArgs &)>;
    using BuildFn = std::function<std::shared_ptr<NodeKernel>(BuildArgs &)>;

    struct NodeSpec
    {
        std::string type;
        std::string category;
        std::string description;
        std::vector<PortSpec> inputs;
        std::vector<PortSpec> outputs;
        std::vector<AttrSpec> attrs;
        NodeRole role = NodeRole::Op;
        InferFn infer;
        ParamsFn params;     // optional; defaults to 0
        BuildFn build;
        std::string source = "builtin";   // or the script that defined it
        std::function<NodePorts(const Node &, const Document &, const Registry &)> dynamicPorts;

        NodePorts Ports(const Node &node, const Document &doc, const Registry &registry) const;
        const AttrSpec *FindAttr(const std::string &name) const;
        int InputIndex(const std::string &name) const;
        int OutputIndex(const std::string &name) const;
        AttrMap Defaults() const;
    };

    using SpecPtr = std::shared_ptr<const NodeSpec>;

    // Thread-safe: the editor hot-reloads script specs while a worker thread
    // may be compiling a model.
    class Registry
    {
    public:
        void Register(NodeSpec spec);
        void Unregister(const std::string &type);
        SpecPtr Find(const std::string &type) const;
        std::vector<SpecPtr> All() const;
        uint64_t Version() const;

    private:
        mutable std::shared_mutex m_mutex;
        std::map<std::string, SpecPtr> m_specs;
        uint64_t m_version = 0;
    };

    void RegisterBuiltins(Registry &registry);

    // Input (or Output) nodes of a graph in port order: by their "index"
    // attribute, then by id.
    std::vector<int> OrderedIo(const Graph &graph, const Registry &registry, NodeRole role);

    // Ports a Group/Repeat node gets from the subgraph it wraps.
    NodePorts SubgraphPorts(const Document &doc, const std::string &subgraph, const Registry &registry);

    // Shared by builtins and scripts.
    std::vector<Dim> ParseShape(const std::string &text, const Variables &vars);
    std::string ShapeError(const std::string &what, const TensorType &got);
}
