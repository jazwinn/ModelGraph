#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Graph/Activity.h"
#include "Graph/Types.h"
#include "NeuralNet/Module.h"
#include "NeuralNet/Transformer/KVCache.h"

namespace mg
{
    // Shape of one KV-cache slot. Each attention (or KVCache node) instance in
    // the compiled model owns one slot; slots may differ in shape.
    struct KVSlotSpec
    {
        size_t kvHeads = 0;
        size_t headDim = 0;
    };

    class GraphCache
    {
    public:
        GraphCache(const std::vector<KVSlotSpec> &slots, size_t batch, size_t maxSeq, tensor::Device device)
            : m_maxSeq(maxSeq)
        {
            m_layers.reserve(slots.size());
            for (const KVSlotSpec &s : slots)
            {
                m_layers.push_back({FTensor({batch, s.kvHeads, maxSeq, s.headDim}, device),
                                    FTensor({batch, s.kvHeads, maxSeq, s.headDim}, device)});
            }
        }

        nn::transformer::KVLayer<Scalar> &Layer(size_t slot)
        {
            if (slot >= m_layers.size()) throw std::out_of_range("GraphCache: slot out of range");
            return m_layers[slot];
        }

        size_t Length() const noexcept { return m_length; }
        size_t MaxLength() const noexcept { return m_maxSeq; }
        void Advance(size_t seq) { m_length += seq; }
        void Reset() noexcept { m_length = 0; }

    private:
        std::vector<nn::transformer::KVLayer<Scalar>> m_layers;
        size_t m_maxSeq;
        size_t m_length = 0;
    };

    struct RunContext
    {
        GraphCache *cache = nullptr;
        size_t startPos = 0;      // absolute position of the first token in this call
        size_t seqLen = 0;
        tensor::Device device = tensor::Device::CPU;
        Activity *activity = nullptr;   // set to record which nodes run
    };

    // The runtime half of a node. Nodes with weights expose them through
    // GetModule(), which the compiler registers under the node's name so the
    // parameter names line up with checkpoints.
    class NodeKernel
    {
    public:
        virtual ~NodeKernel() = default;
        virtual std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &inputs) = 0;
        virtual std::shared_ptr<nn::Module<Scalar>> GetModule() { return nullptr; }
        // Called once every node of a program is built, so a node can refer to
        // another node's module by name (e.g. a tied LM head reusing wte).
        virtual void Resolve(const std::function<std::shared_ptr<nn::Module<Scalar>>(const std::string &)> &) {}
    };

    // A kernel for parameter-free nodes, written as a lambda.
    template <typename F>
    class LambdaKernel : public NodeKernel
    {
    public:
        explicit LambdaKernel(F f) : m_f(std::move(f)) {}
        std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &inputs) override { return m_f(ctx, inputs); }

    private:
        F m_f;
    };

    template <typename F>
    std::shared_ptr<NodeKernel> MakeKernel(F f)
    {
        return std::make_shared<LambdaKernel<F>>(std::move(f));
    }

    // A kernel that is also its own module (its parameters are registered on it).
    class ModuleKernel : public NodeKernel, public nn::Module<Scalar>, public std::enable_shared_from_this<ModuleKernel>
    {
    public:
        std::shared_ptr<nn::Module<Scalar>> GetModule() override { return shared_from_this(); }
    };
}
