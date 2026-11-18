#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Graph/Analyze.h"
#include "Graph/Registry.h"
#include "Graph/Runtime.h"

namespace mg
{
    // A compiled graph. Each node becomes a kernel; kernels with weights are
    // registered as submodules under the node's name, so NamedParameters()
    // yields names like "transformer.h.3.attention.query.weight".
    class Program : public nn::Module<Scalar>
    {
    public:
        struct Source
        {
            int step = -1;   // -1: not connected
            int port = 0;
        };

        struct Step
        {
            std::string name;
            std::string type;
            NodeRole role = NodeRole::Op;
            int ioIndex = 0;                     // Input/Output index
            std::shared_ptr<NodeKernel> kernel;  // null for Input/Output
            std::vector<Source> inputs;
            size_t numOutputs = 0;
        };

        std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &inputs);

        std::vector<Step> steps;
        std::vector<Source> outputs;   // by output index

        // Compiles `graph` (already analysed) into this program.
        void Compile(const Graph &graph, const Analysis &analysis, BuildState &state);

    private:
        std::vector<std::vector<int>> m_uses;   // remaining consumers per step output
    };

    struct BuildOptions
    {
        tensor::Device device = tensor::Device::CPU;
    };

    class GraphModel : public Program
    {
    public:
        // Throws GraphError when the document does not analyse cleanly.
        static std::shared_ptr<GraphModel> Build(const Document &doc, const Registry &registry);

        // ids: [batch, seq] token ids. Returns the first output (logits).
        FTensor Forward(const ITensor &ids, GraphCache *cache = nullptr);

        GraphCache MakeCache(size_t batch, size_t maxSeq) const;
        void ToDevice(tensor::Device device);
        tensor::Device GetDevice() const { return m_device; }

        int64_t ParameterCount() const;

        std::vector<KVSlotSpec> slots;

    private:
        tensor::Device m_device = tensor::Device::CPU;
    };
}
