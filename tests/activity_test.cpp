#include <gtest/gtest.h>

#include "Graph/Activity.h"
#include "Graph/Compiler.h"
#include "Graph/Templates.h"
#include "Graph/Weights.h"

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

    const Node &Named(const Graph &g, const std::string &name)
    {
        for (const Node &n : g.nodes)
            if (n.name == name) return n;
        throw std::runtime_error("no node named " + name);
    }

    struct Tiny
    {
        Document doc = MakeTemplate("tiny");
        std::shared_ptr<GraphModel> model;

        Tiny()
        {
            model = GraphModel::Build(doc, Builtins());
            InitializeFresh(*model, {});
        }

        void Forward(Activity &a)
        {
            ITensor ids({1, 3});
            for (size_t i = 0; i < 3; ++i) ids(0, i) = static_cast<int64_t>(i + 1);
            model->Forward(ids, nullptr, &a);
        }
    };
}

// Keys must match the canvas ids (web/src/canvas/flow.ts rfId): nest ids
// joined by '/', then ':' and the node id. Repeat copies share keys.
TEST(Activity, KeysMatchCanvasIdsThroughNests)
{
    Tiny t;
    Activity a;
    a.SetMode(Activity::Mode::Heat, 0);
    t.Forward(a);
    const Activity::Snapshot s = a.Get();

    const int repeat = Named(t.doc.main, "transformer.h").id;
    const int attention = Named(t.doc.subgraphs.at("Block"), "attention").id;
    const int query = Named(t.doc.subgraphs.at("Attention"), "query").id;
    const int64_t layers = 2;

    const std::string repeatKey = ":" + std::to_string(repeat);
    const std::string attentionKey = std::to_string(repeat) + ":" + std::to_string(attention);
    const std::string queryKey = std::to_string(repeat) + "/" + std::to_string(attention) + ":" + std::to_string(query);

    ASSERT_TRUE(s.nodes.count(repeatKey));
    ASSERT_TRUE(s.nodes.count(attentionKey));
    ASSERT_TRUE(s.nodes.count(queryKey));
    EXPECT_EQ(s.nodes.at(repeatKey).calls, 1u);
    EXPECT_EQ(s.nodes.at(attentionKey).calls, static_cast<uint64_t>(layers));
    EXPECT_EQ(s.nodes.at(queryKey).calls, static_cast<uint64_t>(layers));
    EXPECT_TRUE(s.nodes.at(repeatKey).nest);
    EXPECT_TRUE(s.nodes.at(attentionKey).nest);
    EXPECT_FALSE(s.nodes.at(queryKey).nest);

    // Every main-graph op ran once; inputs and outputs are not recorded.
    for (const Node &n : t.doc.main.nodes)
    {
        const bool io = n.type == "TokenIds" || n.type == "Logits";
        EXPECT_EQ(s.nodes.count(":" + std::to_string(n.id)), io ? 0u : 1u) << n.name;
    }

    EXPECT_EQ(s.forwards, 1u);
    EXPECT_TRUE(s.running.empty());
    double top = 0;
    for (const auto &[key, e] : s.nodes)
        if (key.front() == ':') top += e.seconds;
    EXPECT_LE(top, s.forwardSeconds * 1.0001 + 1e-6);
    EXPECT_GE(s.nodes.at(repeatKey).seconds, s.nodes.at(attentionKey).seconds);
}

TEST(Activity, OffRecordsNothing)
{
    Tiny t;
    Activity a;
    t.Forward(a);
    EXPECT_TRUE(a.Get().nodes.empty());
    EXPECT_EQ(a.Get().forwards, 0u);
}

TEST(Activity, TracePausesAreLeftOutOfTimings)
{
    Tiny t;
    Activity a;
    a.SetMode(Activity::Mode::Trace, 5);
    t.Forward(a);
    const Activity::Snapshot s = a.Get();

    size_t leafCalls = 0;
    for (const auto &[key, e] : s.nodes)
        if (!e.nest) leafCalls += e.calls;
    ASSERT_GT(leafCalls, 10u);
    EXPECT_GE(a.PausedSeconds(), 0.004 * static_cast<double>(leafCalls));
    // The tiny model itself takes far less than the pauses.
    EXPECT_LT(s.forwardSeconds, a.PausedSeconds() / 4);
}

TEST(Activity, StopFlagEndsPausesEarly)
{
    Tiny t;
    Activity a;
    std::atomic<bool> stop{true};
    a.SetStopFlag(&stop);
    a.SetMode(Activity::Mode::Trace, 1000);
    t.Forward(a);
    EXPECT_LT(a.PausedSeconds(), 1.0);
}
