#include "Graph/Document.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace mg
{
    using nlohmann::json;

    // ---- Types helpers -------------------------------------------------------

    const char *DTypeName(DType dtype) { return dtype == DType::F32 ? "f32" : "i64"; }

    DType ParseDType(const std::string &text)
    {
        if (text == "f32") return DType::F32;
        if (text == "i64") return DType::I64;
        throw GraphError("unknown dtype '" + text + "'");
    }

    std::string TensorType::Text() const
    {
        std::string text = "[";
        for (size_t i = 0; i < shape.size(); ++i) text += (i ? "," : "") + shape[i].Text();
        text += "]";
        if (dtype == DType::I64) text += " i64";
        return text;
    }

    std::string AttrText(const AttrValue &value)
    {
        return std::visit([](const auto &v) -> std::string
        {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, bool>) return v ? "true" : "false";
            else if constexpr (std::is_same_v<V, std::string>) return v;
            else if constexpr (std::is_same_v<V, double>)
            {
                std::ostringstream s;
                s << v;
                return s.str();
            }
            else return std::to_string(v);
        }, value);
    }

    const FTensor &AsFloat(const Value &value, const std::string &where)
    {
        if (const auto *t = std::get_if<FTensor>(&value)) return *t;
        throw GraphError(where + ": expected an f32 tensor");
    }

    const ITensor &AsInt(const Value &value, const std::string &where)
    {
        if (const auto *t = std::get_if<ITensor>(&value)) return *t;
        throw GraphError(where + ": expected an i64 tensor");
    }

    // ---- Graph ---------------------------------------------------------------

    Node *Graph::Find(int id)
    {
        for (Node &n : nodes) if (n.id == id) return &n;
        return nullptr;
    }

    const Node *Graph::Find(int id) const
    {
        for (const Node &n : nodes) if (n.id == id) return &n;
        return nullptr;
    }

    const Node *Graph::FindByName(const std::string &name) const
    {
        for (const Node &n : nodes) if (n.name == name) return &n;
        return nullptr;
    }

    Node &Graph::Add(std::string type, std::string name, AttrMap attrs, float x, float y)
    {
        Node n;
        n.id = nextId++;
        n.type = std::move(type);
        n.name = UniqueName(name.empty() ? n.type : name);
        n.attrs = std::move(attrs);
        n.x = x;
        n.y = y;
        nodes.push_back(std::move(n));
        return nodes.back();
    }

    Link &Graph::Connect(int fromNode, const std::string &fromPort, int toNode, const std::string &toPort)
    {
        links.erase(std::remove_if(links.begin(), links.end(), [&](const Link &l)
                                   { return l.to.node == toNode && l.to.port == toPort; }),
                    links.end());
        Link l;
        l.id = nextId++;
        l.from = {fromNode, fromPort};
        l.to = {toNode, toPort};
        links.push_back(std::move(l));
        return links.back();
    }

    void Graph::Remove(int nodeId)
    {
        nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const Node &n) { return n.id == nodeId; }), nodes.end());
        links.erase(std::remove_if(links.begin(), links.end(), [&](const Link &l)
                                   { return l.from.node == nodeId || l.to.node == nodeId; }),
                    links.end());
    }

    void Graph::RemoveLink(int linkId)
    {
        links.erase(std::remove_if(links.begin(), links.end(), [&](const Link &l) { return l.id == linkId; }), links.end());
    }

    const Link *Graph::LinkInto(int node, const std::string &port) const
    {
        for (const Link &l : links) if (l.to.node == node && l.to.port == port) return &l;
        return nullptr;
    }

    std::string Graph::UniqueName(const std::string &stem) const
    {
        if (!FindByName(stem)) return stem;
        for (int i = 1;; ++i)
        {
            std::string candidate = stem + "_" + std::to_string(i);
            if (!FindByName(candidate)) return candidate;
        }
    }

    // ---- JSON ----------------------------------------------------------------

    json AttrToJson(const AttrValue &value)
    {
        return std::visit([](const auto &v) { return json(v); }, value);
    }

    AttrValue AttrFromJson(const json &j)
    {
        if (j.is_boolean()) return j.get<bool>();
        if (j.is_number_integer()) return j.get<int64_t>();
        if (j.is_number_float()) return j.get<double>();
        if (j.is_string()) return j.get<std::string>();
        throw GraphError("attribute values must be bool, number or string, got " + j.dump());
    }

    namespace
    {
        json GraphToJson(const Graph &g)
        {
            json nodes = json::array();
            for (const Node &n : g.nodes)
            {
                json attrs = json::object();
                for (const auto &[k, v] : n.attrs) attrs[k] = AttrToJson(v);
                json jn = {{"id", n.id}, {"type", n.type}, {"name", n.name}, {"attrs", attrs}, {"pos", {n.x, n.y}}};
                if (!n.ui.is_null()) jn["ui"] = n.ui;
                nodes.push_back(std::move(jn));
            }
            json links = json::array();
            for (const Link &l : g.links)
            {
                links.push_back({{"from", {l.from.node, l.from.port}}, {"to", {l.to.node, l.to.port}}});
            }
            return {{"nodes", nodes}, {"links", links}};
        }

        Graph GraphFromJson(const json &j)
        {
            Graph g;
            int maxId = 0;
            for (const json &jn : j.value("nodes", json::array()))
            {
                Node n;
                n.id = jn.at("id").get<int>();
                n.type = jn.at("type").get<std::string>();
                n.name = jn.value("name", n.type + "_" + std::to_string(n.id));
                const json attrs = jn.value("attrs", json::object());
                for (const auto &[k, v] : attrs.items()) n.attrs[k] = AttrFromJson(v);
                if (jn.contains("ui")) n.ui = jn["ui"];
                if (jn.contains("pos"))
                {
                    n.x = jn["pos"][0].get<float>();
                    n.y = jn["pos"][1].get<float>();
                }
                if (g.Find(n.id)) throw GraphError("duplicate node id " + std::to_string(n.id));
                maxId = std::max(maxId, n.id);
                g.nodes.push_back(std::move(n));
            }
            for (const json &jl : j.value("links", json::array()))
            {
                Link l;
                l.from = {jl.at("from")[0].get<int>(), jl.at("from")[1].get<std::string>()};
                l.to = {jl.at("to")[0].get<int>(), jl.at("to")[1].get<std::string>()};
                g.links.push_back(std::move(l));
            }
            g.nextId = maxId + 1;
            for (Link &l : g.links) l.id = g.nextId++;
            return g;
        }
    }

    Variables Document::NumericVariables() const
    {
        Variables vars;
        // Variables may reference earlier ones ("head_dim": "$d_model / $num_heads"),
        // so resolve iteratively until nothing changes.
        std::map<std::string, std::string> pending;
        for (const auto &[k, v] : variables)
        {
            if (const auto *b = std::get_if<bool>(&v)) vars[k] = *b ? 1.0 : 0.0;
            else if (const auto *i = std::get_if<int64_t>(&v)) vars[k] = static_cast<double>(*i);
            else if (const auto *d = std::get_if<double>(&v)) vars[k] = *d;
            else pending[k] = std::get<std::string>(v);
        }
        bool progress = true;
        while (!pending.empty() && progress)
        {
            progress = false;
            for (auto it = pending.begin(); it != pending.end();)
            {
                try
                {
                    vars[it->first] = Evaluate(it->second, vars);
                    it = pending.erase(it);
                    progress = true;
                }
                catch (const std::exception &)
                {
                    ++it;
                }
            }
        }
        return vars;
    }

    std::filesystem::path Document::Resolve(const std::string &relative) const
    {
        std::filesystem::path p(relative);
        if (p.empty() || p.is_absolute() || path.empty()) return p;
        return (path.parent_path() / p).lexically_normal();
    }

    json Document::ToJson() const
    {
        json vars = json::object();
        for (const auto &[k, v] : variables) vars[k] = AttrToJson(v);

        json subs = json::object();
        for (const auto &[k, g] : subgraphs) subs[k] = GraphToJson(g);

        json rules = json::array();
        for (const auto &r : model.rules) rules.push_back({{"from", r.from}, {"to", r.to}});

        json jmodel = {
            {"checkpoint", model.checkpoint},
            {"format", model.format},
            {"rename_rules", rules},
            {"tokenizer", model.tokenizer},
            {"chat_template", model.chatTemplate},
            {"system_prompt", model.systemPrompt},
            {"eos", model.eos},
            {"device", model.device},
            {"context_size", model.contextSize},
        };

        return {
            {"format", "modelgraph"},
            {"version", 1},
            {"name", name},
            {"variables", vars},
            {"model", jmodel},
            {"graph", GraphToJson(main)},
            {"subgraphs", subs},
        };
    }

    Document Document::FromJson(const json &j)
    {
        Document d;
        d.name = j.value("name", "Untitled");
        // Keep the json::value() results alive: items() only references them.
        const json variables = j.value("variables", json::object());
        for (const auto &[k, v] : variables.items()) d.variables[k] = AttrFromJson(v);
        d.main = GraphFromJson(j.value("graph", json::object()));
        const json subgraphs = j.value("subgraphs", json::object());
        for (const auto &[k, v] : subgraphs.items()) d.subgraphs[k] = GraphFromJson(v);

        const json m = j.value("model", json::object());
        d.model.checkpoint = m.value("checkpoint", "");
        d.model.format = m.value("format", "raw");
        for (const json &r : m.value("rename_rules", json::array()))
        {
            d.model.rules.push_back({r.at("from").get<std::string>(), r.at("to").get<std::string>()});
        }
        d.model.tokenizer = m.value("tokenizer", "");
        d.model.chatTemplate = m.value("chat_template", "");
        d.model.systemPrompt = m.value("system_prompt", d.model.systemPrompt);
        d.model.eos = m.value("eos", std::vector<int64_t>{});
        d.model.device = m.value("device", "cuda");
        d.model.contextSize = m.value("context_size", int64_t{2048});
        return d;
    }

    Document Document::Load(const std::filesystem::path &file)
    {
        std::ifstream in(file);
        if (!in) throw GraphError("cannot open '" + file.string() + "'");
        json j;
        try
        {
            in >> j;
        }
        catch (const std::exception &e)
        {
            throw GraphError("'" + file.string() + "' is not valid JSON: " + e.what());
        }
        Document d = FromJson(j);
        d.path = std::filesystem::absolute(file);
        return d;
    }

    void Document::Save(const std::filesystem::path &file)
    {
        std::ofstream out(file);
        if (!out) throw GraphError("cannot write '" + file.string() + "'");
        out << ToJson().dump(2) << "\n";
        path = std::filesystem::absolute(file);
    }
}
