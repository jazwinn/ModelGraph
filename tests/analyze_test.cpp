#include <gtest/gtest.h>

#include "Graph/Analyze.h"
#include "Graph/Compiler.h"
#include "Graph/Templates.h"
#include "Models/ModelConfigs.h"

using namespace mg;

namespace
{
    Registry &Builtins()
    {
        static Registry r;
        static const bool once = (RegisterBuiltins(r), true);
        (void)once;
        return r;
    }

    int64_t CausalModelParams(const nn::transformer::config::TransformerConfig &cfg)
    {
        // Closed form for the reference architecture.
        const int64_t d = cfg.d_model, f = cfg.ffn_dim, v = cfg.vocab_size, L = cfg.num_layers;
        const int64_t q = cfg.num_heads * cfg.HeadDim(), kv = cfg.KVHeads() * cfg.HeadDim();
        const bool gated = cfg.mlp_type == nn::transformer::config::MLPType::Gated;
        const int64_t norm = cfg.norm_type == nn::transformer::config::NormType::RMSNorm ? d : 2 * d;
        int64_t attn = d * (q + 2 * kv) + q * d + (cfg.attention_bias ? q + 2 * kv : 0) + (cfg.output_bias ? d : 0);
        int64_t mlp = (gated ? 3 : 2) * d * f + (cfg.output_bias ? (gated ? 2 * f : f) + d : 0);
        int64_t total = v * d + L * (attn + mlp + 2 * norm) + norm;
        if (cfg.position == nn::transformer::config::PositionEncoding::Learned) total += cfg.max_seq_len * d;
        if (!cfg.tie_word_embeddings) total += d * v;
        return total;
    }
}

TEST(Analyze, Gpt2TemplateHasExpectedShapesAndParams)
{
    const Document doc = MakeTemplate("gpt2");
    const Analysis a = Analyze(doc, doc.main, Builtins());
    ASSERT_TRUE(a.Ok()) << a.Summary();
    ASSERT_EQ(a.results.size(), 1u);
    EXPECT_EQ(a.results[0].Text(), "[B,S,50257]");
    EXPECT_EQ(a.totalParams, CausalModelParams(models::gpt2::Small()));
    EXPECT_EQ(a.totalParams, 124439808);   // GPT-2 small with tied head
}

TEST(Analyze, LlamaTemplateParams)
{
    const Document doc = MakeTemplate("llama3.2-1b");
    const Analysis a = Analyze(doc, doc.main, Builtins());
    ASSERT_TRUE(a.Ok()) << a.Summary();
    EXPECT_EQ(a.totalParams, CausalModelParams(models::llama::Llama32_1B()));
}

TEST(Analyze, ReportsShapeMismatchOnTheOffendingNode)
{
    Document doc = MakeTemplate("tiny");
    Node *up = nullptr;
    for (Node &n : doc.subgraphs["MLP"].nodes) if (n.name == "up") up = &n;
    ASSERT_NE(up, nullptr);
    up->attrs["in_features"] = int64_t{65};

    const Analysis sub = Analyze(doc, doc.subgraphs["MLP"], Builtins());
    ASSERT_TRUE(sub.errors.contains(up->id));
    EXPECT_NE(sub.errors.at(up->id).find("needs size 65"), std::string::npos) << sub.errors.at(up->id);

    // The error surfaces through every nest that contains it: MLP -> Block -> main.
    EXPECT_FALSE(Analyze(doc, doc.subgraphs["Block"], Builtins()).Ok());
    EXPECT_FALSE(Analyze(doc, doc.main, Builtins()).Ok());
}

TEST(Analyze, DetectsCyclesAndMissingInputs)
{
    Document doc;
    Graph &g = doc.main;
    const int a = g.Add("Add", "a").id;
    const int b = g.Add("Add", "b").id;
    g.Connect(a, "out", b, "a");
    g.Connect(b, "out", a, "a");
    const int lone = g.Add("Sqrt", "lone").id;
    const Analysis an = Analyze(doc, g, Builtins());
    EXPECT_NE(an.errors.at(a).find("cycle"), std::string::npos);
    EXPECT_NE(an.errors.at(lone).find("not connected"), std::string::npos);
}

TEST(Analyze, ReshapeTracksSymbols)
{
    Document doc;
    doc.variables["d_model"] = int64_t{64};
    Graph &g = doc.main;
    const int in = g.Add("Input", "x", {{"shape", std::string("B, S, $d_model")}}).id;
    const int r1 = g.Add("Reshape", "r1", {{"shape", std::string("B, S, 4, -1")}}).id;
    const int r2 = g.Add("Reshape", "r2", {{"shape", std::string("-1, 64")}}).id;
    g.Connect(in, "out", r1, "x");
    g.Connect(r1, "y", r2, "x");
    const Analysis a = Analyze(doc, g, Builtins());
    ASSERT_TRUE(a.Ok()) << a.Summary();
    EXPECT_EQ(a.outputs.at(r1)[0].Text(), "[B,S,4,16]");
    EXPECT_EQ(a.outputs.at(r2)[0].Text(), "[?,64]");
}

TEST(Nests, GroupPortsComeFromTheSubgraph)
{
    Document doc;
    doc.variables["d_model"] = int64_t{8};
    Graph &sub = doc.subgraphs["Gate"];
    const int x = sub.Add("Input", "x", {{"index", int64_t{0}}, {"shape", std::string("B, S, $d_model")}}).id;
    const int g = sub.Add("Input", "g", {{"index", int64_t{1}}, {"shape", std::string("B, S, $d_model")}}).id;
    const int lin = sub.Add("Linear", "proj").id;
    const int mul = sub.Add("Mul", "mul").id;
    const int y = sub.Add("Output", "y").id;
    sub.Connect(x, "out", lin, "x");
    sub.Connect(lin, "y", mul, "a");
    sub.Connect(g, "out", mul, "b");
    sub.Connect(mul, "out", y, "value");

    Graph &m = doc.main;
    const int a = m.Add("Input", "a", {{"shape", std::string("B, S, $d_model")}}).id;
    const int box = m.Add("Group", "gate", {{"subgraph", std::string("Gate")}}).id;
    const int out = m.Add("Output", "out").id;
    m.Connect(a, "out", box, "x");
    m.Connect(a, "out", box, "g");
    m.Connect(box, "y", out, "value");

    const Analysis an = Analyze(doc, m, Builtins());
    ASSERT_TRUE(an.Ok()) << an.Summary();
    ASSERT_EQ(an.ports.at(box).inputs.size(), 2u);
    EXPECT_EQ(an.ports.at(box).inputs[1].name, "g");
    EXPECT_EQ(an.outputs.at(box)[0].Text(), "[B,S,8]");
    EXPECT_EQ(an.params.at(box), 8 * 8 + 8);

    // Weights are named after the Group node.
    auto program = std::make_shared<Program>();
    BuildState state{doc, Builtins(), {}, {}, {}};
    program->Compile(m, an, state);
    EXPECT_EQ(program->NamedParameters()[0].name, "gate.proj.weight");
}

TEST(Nests, RepeatCarriesOutputZeroAndBroadcastsOtherInputs)
{
    Document doc;
    doc.variables["d_model"] = int64_t{4};
    Graph &sub = doc.subgraphs["Step"];
    const int x = sub.Add("Input", "x", {{"index", int64_t{0}}, {"shape", std::string("B, $d_model")}}).id;
    const int d = sub.Add("Input", "delta", {{"index", int64_t{1}}, {"shape", std::string("B, $d_model")}}).id;
    const int add = sub.Add("Add", "add").id;
    const int y = sub.Add("Output", "y").id;
    sub.Connect(x, "out", add, "a");
    sub.Connect(d, "out", add, "b");
    sub.Connect(add, "out", y, "value");

    Graph &m = doc.main;
    const int a = m.Add("Input", "a", {{"index", int64_t{0}}, {"shape", std::string("B, $d_model")}}).id;
    const int b = m.Add("Input", "b", {{"index", int64_t{1}}, {"shape", std::string("B, $d_model")}}).id;
    const int rep = m.Add("Repeat", "steps", {{"subgraph", std::string("Step")}, {"count", int64_t{3}}}).id;
    const int out = m.Add("Output", "out").id;
    m.Connect(a, "out", rep, "x");
    m.Connect(b, "out", rep, "delta");
    m.Connect(rep, "y", out, "value");

    const Analysis an = Analyze(doc, m, Builtins());
    ASSERT_TRUE(an.Ok()) << an.Summary();
    auto program = std::make_shared<Program>();
    BuildState state{doc, Builtins(), {}, {}, {}};
    program->Compile(m, an, state);
    RunContext ctx;
    const FTensor r = AsFloat(program->Run(ctx, {FTensor({1, 4}, 1.0f), FTensor({1, 4}, 2.0f)}).at(0), "r");
    EXPECT_FLOAT_EQ(r.Data()[0], 7.0f);   // 1 + 3 * 2
}
