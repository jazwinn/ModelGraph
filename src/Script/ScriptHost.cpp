#include "Script/ScriptHost.h"

#include <algorithm>
#include <fstream>
#include <set>

namespace mg::script
{
    namespace fs = std::filesystem;

    void ScriptHost::AddDirectory(const fs::path &dir)
    {
        const fs::path abs = fs::absolute(dir).lexically_normal();
        if (std::find(m_dirs.begin(), m_dirs.end(), abs) == m_dirs.end()) m_dirs.push_back(abs);
    }

    void ScriptHost::ClearDirectories()
    {
        m_dirs.clear();
    }

    bool ScriptHost::Load(const fs::path &file, Entry &entry)
    {
        std::shared_ptr<ScriptState> state;
        try
        {
            state = LoadScript(file);
        }
        catch (const std::exception &e)
        {
            entry.status.ok = false;
            entry.status.error = e.what();
            return false;   // keep the previous version registered
        }

        const std::string type = ScriptType(*state);
        const SpecPtr existing = m_registry.Find(type);
        if (existing && existing->source != file.string())
        {
            entry.status.ok = false;
            entry.status.error = "node type '" + type + "' is already defined by " +
                                 (existing->source == "builtin" ? std::string("a built-in node") : existing->source);
            return false;
        }

        bool changed = false;
        if (!entry.status.type.empty() && entry.status.type != type)
        {
            m_registry.Unregister(entry.status.type);   // the script was renamed
            changed = true;
        }
        if (!entry.slot) entry.slot = std::make_shared<ScriptSlot>();
        entry.slot->Set(state);
        m_registry.Register(MakeSpec(entry.slot, file));
        entry.status.type = type;
        entry.status.ok = true;
        entry.status.error.clear();
        ++entry.status.reloads;
        (void)changed;
        return true;
    }

    bool ScriptHost::Poll()
    {
        bool changed = false;
        std::set<fs::path> seen;
        for (const fs::path &dir : m_dirs)
        {
            std::error_code ec;
            if (!fs::is_directory(dir, ec)) continue;
            for (const auto &item : fs::directory_iterator(dir, ec))
            {
                if (!item.is_regular_file() || item.path().extension() != ".py") continue;
                if (item.path().filename().string().starts_with("_")) continue;   // helpers, __init__.py
                const fs::path file = item.path().lexically_normal();
                seen.insert(file);
                const auto mtime = fs::last_write_time(file, ec);
                if (ec) continue;

                auto it = m_entries.find(file);
                if (it == m_entries.end())
                {
                    Entry e;
                    e.mtime = mtime;
                    e.status.file = file;
                    it = m_entries.emplace(file, std::move(e)).first;
                    changed |= Load(file, it->second);
                }
                else if (it->second.mtime != mtime)
                {
                    it->second.mtime = mtime;
                    changed |= Load(file, it->second);
                }
            }
        }

        for (auto it = m_entries.begin(); it != m_entries.end();)
        {
            if (seen.contains(it->first))
            {
                ++it;
                continue;
            }
            if (!it->second.status.type.empty())
            {
                const SpecPtr spec = m_registry.Find(it->second.status.type);
                if (spec && spec->source == it->first.string()) m_registry.Unregister(it->second.status.type);
                changed = true;
            }
            it = m_entries.erase(it);
        }
        return changed;
    }

    std::vector<ScriptStatus> ScriptHost::Statuses() const
    {
        std::vector<ScriptStatus> out;
        for (const auto &[file, e] : m_entries) out.push_back(e.status);
        return out;
    }

    fs::path ScriptHost::CreateStarter(const fs::path &dir, const std::string &name)
    {
        fs::create_directories(dir);
        fs::path file = dir / (name + ".py");
        for (int i = 2; fs::exists(file); ++i) file = dir / (name + std::to_string(i) + ".py");
        std::ofstream out(file);
        out << R"PY(# A custom node. Save this file and ModelGraph reloads it immediately:
# models that are already built pick up a new forward() on the next token.
#
# Tensors are InferenceEngine tensors (CPU or CUDA). Available:
#   x.shape, x.reshape(...), x.transpose(a, b), x.slice(axis, start, n),
#   x.unsqueeze(a), x.squeeze(a), x.mean(axis, keepdim), x.numpy(),
#   + - * / @ with tensors or numbers,
#   ie.matmul, ie.linear(x, w, b), ie.softmax(x, axis), ie.silu/gelu/relu/sigmoid/tanh,
#   ie.sqrt, ie.sdpa(q, k, v, causal, scale), ie.split_heads, ie.merge_heads,
#   ie.zeros(shape), ie.full(shape, v), ie.tensor(numpy_array),
#   and CPU fallbacks ie.exp/log/abs/pow/clamp/map.
# Axes are 0-based; negative axes count from the end.
import ie

name = ")PY" << name << R"PY("
category = "Custom"
description = "y = x * sigmoid(beta * x)"

inputs = ["x"]
outputs = ["y"]

# bool/int/float/str values become node attributes. Strings starting with $
# are expressions over the document variables, e.g. "$d_model".
attrs = {"beta": 1.0}

# Optional: learned parameters (loaded from checkpoints like any other weight).
# def params(a):
#     return {"weight": {"shape": [a.dim], "init": "ones"}}

# Optional: output shape from input shapes (default: same as the first input).
# def shape(a, x):
#     return x


def forward(ctx, x):
    return x * ie.sigmoid(x * ctx.attrs.beta)
)PY";
        return file;
    }
}
