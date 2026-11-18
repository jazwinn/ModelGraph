// The graph compiler must reproduce InferenceEngine's hand-written
// models::CausalModel exactly: same parameter names, same logits, with and
// without the KV cache.

#include <gtest/gtest.h>

#include <map>
#include <random>

#include "Graph/Compiler.h"
#include "Graph/Templates.h"
#include "Graph/Weights.h"
#include "Models/CausalModel.h"

using namespace mg;
namespace cfgns = nn::transformer::config;

namespace
{
    Registry &Builtins()
    {
        static Registry r;
        static const bool once = (RegisterBuiltins(r), true);
        (void)once;
        return r;
    }

    void Randomize(nn::Module<Scalar> &m, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> normal(0.0f, 0.1f);
        for (const auto &[name, p] : m.NamedParameters())
        {
            FTensor t(p->Shape());
            for (size_t i = 0; i < t.Size(); ++i) t.Data()[i] = normal(rng) + (p->Rank() == 1 ? 1.0f : 0.0f);
            *p = t;
        }
    }

    float MaxDiff(const FTensor &a, const FTensor &b)
    {
        EXPECT_EQ(a.Shape(), b.Shape());
        const FTensor x = a.To(tensor::Device::CPU).Contiguous(), y = b.To(tensor::Device::CPU).Contiguous();
        float m = 0;
        for (size_t i = 0; i < x.Size(); ++i) m = std::max(m, std::abs(x.Data()[i] - y.Data()[i]));
        return m;
    }

    ITensor Ids(std::vector<int64_t> v)
    {
        ITensor t({1, v.size()});
        for (size_t i = 0; i < v.size(); ++i) t(0, i) = v[i];
        return t;
    }

    cfgns::TransformerConfig Base()
    {
        cfgns::TransformerConfig c;
        c.vocab_size = 40; c.d_model = 32; c.num_layers = 2; c.num_heads = 4; c.ffn_dim = 48; c.max_seq_len = 64;
        return c;
    }

    void ExpectEquivalent(const cfgns::TransformerConfig &cfg, tensor::Device device = tensor::Device::CPU)
    {
        models::CausalModel<Scalar> ref(cfg);
        Randomize(ref, 7);

        const Document doc = FromTransformerConfig(cfg, "test");
        std::shared_ptr<GraphModel> graph = GraphModel::Build(doc, Builtins());

        std::map<std::string, std::shared_ptr<FTensor>> refParams;
        for (const auto &[n, p] : ref.NamedParameters()) refParams[n] = p;
        std::map<std::string, std::shared_ptr<FTensor>> graphParams;
        for (const auto &[n, p] : graph->NamedParameters()) graphParams[n] = p;

        ASSERT_EQ(refParams.size(), graphParams.size());
        for (const auto &[n, p] : graphParams)
        {
            ASSERT_TRUE(refParams.contains(n)) << "graph parameter '" << n << "' has no CausalModel counterpart";
            ASSERT_EQ(refParams[n]->Shape(), p->Shape()) << n;
        }
        ASSERT_EQ(CarryOver(ref, *graph).size(), refParams.size());

        ref.To(device);
        graph->ToDevice(device);

        const std::vector<int64_t> prompt = {3, 17, 5, 22, 9, 1};
        const FTensor a = ref.Forward(Ids(prompt).To(device));
        const FTensor b = graph->Forward(Ids(prompt));
        EXPECT_LT(MaxDiff(a, b), 1e-4f);

        // Prefill four tokens, then decode the rest one at a time through the cache.
        auto refCache = ref.MakeCache(1, 16);
        GraphCache graphCache = graph->MakeCache(1, 16);
        const std::vector<int64_t> head(prompt.begin(), prompt.begin() + 4);
        EXPECT_LT(MaxDiff(ref.Forward(Ids(head).To(device), &refCache), graph->Forward(Ids(head), &graphCache)), 1e-4f);
        for (size_t i = 4; i < prompt.size(); ++i)
        {
            const FTensor r = ref.Forward(Ids({prompt[i]}).To(device), &refCache);
            const FTensor g = graph->Forward(Ids({prompt[i]}), &graphCache);
            EXPECT_LT(MaxDiff(r, g), 1e-4f) << "decode step " << i;
            // ...and it must agree with the uncached full pass at that position.
            EXPECT_LT(MaxDiff(g.Row(0).Row(0), b.Row(0).Row(i)), 1e-4f);
        }
    }
}

TEST(Equivalence, Gpt2Style)
{
    cfgns::TransformerConfig c = Base();
    c.activation = tensor::Activation::GELUTanh;
    ExpectEquivalent(c);
}

TEST(Equivalence, LlamaStyle)
{
    cfgns::TransformerConfig c = Base();
    c.num_kv_heads = 2;
    c.norm_type = cfgns::NormType::RMSNorm;
    c.mlp_type = cfgns::MLPType::Gated;
    c.activation = tensor::Activation::SiLU;
    c.position = cfgns::PositionEncoding::Rotary;
    c.rope_scaling.type = cfgns::RopeScalingType::Llama3;
    c.rope_scaling.factor = 8;
    c.rope_scaling.original_max_position = 32;
    c.attention_bias = false;
    c.output_bias = false;
    ExpectEquivalent(c);
}

TEST(Equivalence, PostNormUntiedHead)
{
    cfgns::TransformerConfig c = Base();
    c.norm_placement = cfgns::NormPlacement::Post;
    c.tie_word_embeddings = false;
    c.position = cfgns::PositionEncoding::None;
    ExpectEquivalent(c);
}

#ifdef IE_ENABLE_CUDA
TEST(Equivalence, LlamaStyleOnCuda)
{
    cfgns::TransformerConfig c = Base();
    c.num_kv_heads = 2;
    c.norm_type = cfgns::NormType::RMSNorm;
    c.mlp_type = cfgns::MLPType::Gated;
    c.activation = tensor::Activation::SiLU;
    c.position = cfgns::PositionEncoding::Rotary;
    c.attention_bias = false;
    c.output_bias = false;
    ExpectEquivalent(c, tensor::Device::CUDA);
}
#endif

// Attention rebuilt from parts (Linear, SplitHeads, RoPE, KVCache, SDPA,
// MergeHeads) must match the fused Attention node.
TEST(Equivalence, DecomposedAttentionMatchesFusedNode)
{
    Document doc;
    doc.variables = {{"d_model", int64_t{32}}, {"num_heads", int64_t{4}}, {"max_seq_len", int64_t{64}}};

    auto build = [&](bool decomposed)
    {
        Document d = doc;
        Graph &g = d.main;
        const int in = g.Add("Input", "x", {{"shape", std::string("B, S, $d_model")}}).id;
        int last;
        if (!decomposed)
        {
            last = g.Add("Attention", "attention", {{"num_kv_heads", int64_t{2}}, {"rope", true}, {"qkv_bias", false}}).id;
            g.Connect(in, "out", last, "x");
        }
        else
        {
            const AttrMap noBias = {{"bias", false}};
            const int q = g.Add("Linear", "attention.query", noBias).id;
            const int k = g.Add("Linear", "attention.key", {{"bias", false}, {"out_features", int64_t{16}}}).id;
            const int v = g.Add("Linear", "attention.value", {{"bias", false}, {"out_features", int64_t{16}}}).id;
            const int o = g.Add("Linear", "attention.output_proj").id;
            const int sq = g.Add("SplitHeads", "sq", {{"head_dim", int64_t{8}}}).id;
            const int sk = g.Add("SplitHeads", "sk", {{"num_heads", int64_t{2}}, {"head_dim", int64_t{8}}}).id;
            const int sv = g.Add("SplitHeads", "sv", {{"num_heads", int64_t{2}}, {"head_dim", int64_t{8}}}).id;
            const int rq = g.Add("RoPE", "rq").id;
            const int rk = g.Add("RoPE", "rk").id;
            const int kv = g.Add("KVCache", "cache").id;
            const int at = g.Add("SDPA", "sdpa").id;
            const int mh = g.Add("MergeHeads", "merge").id;
            for (int lin : {q, k, v}) g.Connect(in, "out", lin, "x");
            g.Connect(q, "y", sq, "x");
            g.Connect(k, "y", sk, "x");
            g.Connect(v, "y", sv, "x");
            g.Connect(sq, "heads", rq, "x");
            g.Connect(sk, "heads", rk, "x");
            g.Connect(rk, "y", kv, "k");
            g.Connect(sv, "heads", kv, "v");
            g.Connect(rq, "y", at, "q");
            g.Connect(kv, "k_all", at, "k");
            g.Connect(kv, "v_all", at, "v");
            g.Connect(at, "out", mh, "heads");
            g.Connect(mh, "x", o, "x");
            last = o;
        }
        const int out = g.Add("Output", "y").id;
        g.Connect(last, decomposed ? "y" : "y", out, "value");

        const Analysis a = Analyze(d, g, Builtins());
        EXPECT_TRUE(a.Ok()) << a.Summary();
        auto p = std::make_shared<Program>();
        BuildState state{d, Builtins(), {}, {}, {}};
        p->Compile(g, a, state);
        return std::make_pair(p, state.slots);
    };

    auto [fused, fusedSlots] = build(false);
    auto [parts, partSlots] = build(true);
    Randomize(*fused, 3);
    ASSERT_EQ(CarryOver(*fused, *parts).size(), 5u);   // q, k, v, out weights + out bias

    std::mt19937_64 rng(5);
    std::normal_distribution<float> normal;
    FTensor x({1, 5, 32});
    for (size_t i = 0; i < x.Size(); ++i) x.Data()[i] = normal(rng);

    GraphCache c1(fusedSlots, 1, 16, tensor::Device::CPU), c2(partSlots, 1, 16, tensor::Device::CPU);
    for (size_t start = 0; start < 5;)
    {
        const size_t n = start == 0 ? 3 : 1;   // prefill 3, then 1 at a time
        RunContext ctx1{&c1, c1.Length(), n, tensor::Device::CPU};
        RunContext ctx2{&c2, c2.Length(), n, tensor::Device::CPU};
        const FTensor chunk = x.Slice(1, start, n).Contiguous();
        const FTensor y1 = AsFloat(fused->Run(ctx1, {chunk}).at(0), "y1");
        const FTensor y2 = AsFloat(parts->Run(ctx2, {chunk}).at(0), "y2");
        EXPECT_LT(MaxDiff(y1, y2), 1e-4f) << "at position " << start;
        c1.Advance(n);
        c2.Advance(n);
        start += n;
    }
}
