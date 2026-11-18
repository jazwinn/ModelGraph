#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "Graph/Compiler.h"
#include "Graph/Templates.h"
#include "Script/ScriptHost.h"

using namespace mg;
namespace fs = std::filesystem;

namespace
{
    fs::path FreshDir(const std::string &name)
    {
        const fs::path dir = fs::path(MG_TEST_TMP) / name;
        fs::remove_all(dir);
        fs::create_directories(dir);
        return dir;
    }

    void Write(const fs::path &file, const std::string &text)
    {
        // Bump the timestamp explicitly so back-to-back writes register as changes.
        const bool existed = fs::exists(file);
        const auto before = existed ? fs::last_write_time(file) : fs::file_time_type{};
        std::ofstream(file) << text;
        if (existed && fs::last_write_time(file) <= before) fs::last_write_time(file, before + std::chrono::seconds(1));
    }

    const char *kScale = R"(
name = "TestScale"
attrs = {"factor": 2.0}

def forward(ctx, x):
    return x * ctx.attrs.factor
)";

    // One input -> TestScale -> output, as a document.
    Document ScaleGraph()
    {
        Document d;
        d.variables["d_model"] = int64_t{4};
        Graph &g = d.main;
        const int in = g.Add("Input", "x", {{"shape", std::string("B, $d_model")}}).id;
        const int s = g.Add("TestScale", "scale").id;
        const int out = g.Add("Output", "y").id;
        g.Connect(in, "out", s, "x");
        g.Connect(s, "y", out, "value");
        return d;
    }

    std::shared_ptr<Program> Compile(const Document &d, const Registry &r)
    {
        const Analysis a = Analyze(d, d.main, r);
        EXPECT_TRUE(a.Ok()) << a.Summary();
        auto p = std::make_shared<Program>();
        BuildState state{d, r, {}, {}, {}};
        p->Compile(d.main, a, state);
        return p;
    }

    float RunFirst(Program &p, float value)
    {
        RunContext ctx;
        FTensor x({1, 4}, value);
        return AsFloat(p.Run(ctx, {x}).at(0), "y").Data()[0];
    }
}

TEST(Script, LoadsRunsAndHotReloadsForward)
{
    const fs::path dir = FreshDir("hot");
    Write(dir / "scale.py", kScale);

    Registry registry;
    RegisterBuiltins(registry);
    script::ScriptHost host(registry);
    host.AddDirectory(dir);
    EXPECT_TRUE(host.Poll());
    ASSERT_TRUE(registry.Find("TestScale"));

    auto program = Compile(ScaleGraph(), registry);
    EXPECT_FLOAT_EQ(RunFirst(*program, 3.0f), 6.0f);

    // Edit the script: the already-built program uses the new forward().
    Write(dir / "scale.py", R"(
name = "TestScale"
attrs = {"factor": 2.0}

def forward(ctx, x):
    return x * ctx.attrs.factor + 1
)");
    EXPECT_TRUE(host.Poll());
    EXPECT_FLOAT_EQ(RunFirst(*program, 3.0f), 7.0f);

    // A broken edit keeps the last good version running and reports the error.
    Write(dir / "scale.py", "name = 'TestScale'\ndef forward(ctx, x)\n");
    host.Poll();
    EXPECT_FALSE(host.Statuses().at(0).ok);
    EXPECT_FLOAT_EQ(RunFirst(*program, 3.0f), 7.0f);

    // Deleting the file removes the node type.
    fs::remove(dir / "scale.py");
    EXPECT_TRUE(host.Poll());
    EXPECT_FALSE(registry.Find("TestScale"));
}

TEST(Script, ParamsShapesAndAttributeExpressions)
{
    const fs::path dir = FreshDir("params");
    Write(dir / "affine.py", R"(
import ie
name = "TestAffine"
attrs = {"dim": "$d_model", "out": 3}

def params(a):
    return {"w": [a.out, a.dim], "b": {"shape": [a.out], "init": "ones"}}

def shape(a, x):
    return (x[0], a.out)

def forward(ctx, x):
    return ie.linear(x, ctx.params.w, ctx.params.b)
)");
    Registry registry;
    RegisterBuiltins(registry);
    script::ScriptHost host(registry);
    host.AddDirectory(dir);
    host.Poll();
    ASSERT_TRUE(registry.Find("TestAffine")) << host.Statuses().at(0).error;

    Document d = ScaleGraph();
    d.main.Find(2)->type = "TestAffine";
    const Analysis a = Analyze(d, d.main, registry);
    ASSERT_TRUE(a.Ok()) << a.Summary();
    EXPECT_EQ(a.outputs.at(2)[0].Text(), "[B,3]");
    EXPECT_EQ(a.params.at(2), 3 * 4 + 3);

    auto program = Compile(d, registry);
    ASSERT_EQ(program->NamedParameters().size(), 2u);
    EXPECT_EQ(program->NamedParameters()[0].name, "scale.b");
}

TEST(Script, ShapeErrorsSurfaceInAnalysis)
{
    const fs::path dir = FreshDir("shape");
    Write(dir / "bad.py", R"(
name = "TestBad"

def shape(a, x):
    raise ValueError("needs rank 3")

def forward(ctx, x):
    return x
)");
    Registry registry;
    RegisterBuiltins(registry);
    script::ScriptHost host(registry);
    host.AddDirectory(dir);
    host.Poll();

    Document d = ScaleGraph();
    d.main.Find(2)->type = "TestBad";
    const Analysis a = Analyze(d, d.main, registry);
    ASSERT_TRUE(a.errors.contains(2));
    EXPECT_NE(a.errors.at(2).find("needs rank 3"), std::string::npos);
}

TEST(Script, NumpyInteropAndMultipleOutputs)
{
    const fs::path dir = FreshDir("numpy");
    Write(dir / "split.py", R"(
import ie
import numpy as np
name = "TestSplit"
outputs = ["lo", "hi"]

def shape(a, x):
    return [x, x]

def forward(ctx, x):
    arr = x.numpy()
    return ie.tensor(np.minimum(arr, 0)), ie.tensor(np.maximum(arr, 0))
)");
    Registry registry;
    RegisterBuiltins(registry);
    script::ScriptHost host(registry);
    host.AddDirectory(dir);
    host.Poll();
    ASSERT_TRUE(registry.Find("TestSplit")) << host.Statuses().at(0).error;

    Document d = ScaleGraph();
    d.main.Find(2)->type = "TestSplit";
    d.main.links.back().from.port = "hi";
    auto program = Compile(d, registry);
    EXPECT_FLOAT_EQ(RunFirst(*program, -2.0f), 0.0f);
    EXPECT_FLOAT_EQ(RunFirst(*program, 5.0f), 5.0f);
}

TEST(Script, StarterScriptLoads)
{
    const fs::path dir = FreshDir("starter");
    script::ScriptHost::CreateStarter(dir, "Swish");
    Registry registry;
    RegisterBuiltins(registry);
    script::ScriptHost host(registry);
    host.AddDirectory(dir);
    host.Poll();
    ASSERT_TRUE(registry.Find("Swish")) << host.Statuses().at(0).error;
}
