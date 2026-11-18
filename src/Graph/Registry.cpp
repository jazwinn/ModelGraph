#include "Graph/Registry.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>

namespace mg
{
    // ---- Attrs ---------------------------------------------------------------

    AttrValue Attrs::Raw(const std::string &name) const
    {
        const auto it = m_values.find(name);
        if (it != m_values.end()) return it->second;
        if (const AttrSpec *spec = m_spec.FindAttr(name)) return spec->def;
        throw GraphError("node type '" + m_spec.type + "' has no attribute '" + name + "'");
    }

    double Attrs::Float(const std::string &name) const
    {
        const AttrValue v = Raw(name);
        if (const auto *b = std::get_if<bool>(&v)) return *b ? 1.0 : 0.0;
        if (const auto *i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
        if (const auto *d = std::get_if<double>(&v)) return *d;
        const std::string &expr = std::get<std::string>(v);
        try
        {
            return Evaluate(expr, m_vars);
        }
        catch (const UnknownSymbol &e)
        {
            throw GraphError("attribute '" + name + "': " + e.what() + " (define it under Variables)");
        }
        catch (const std::exception &e)
        {
            throw GraphError("attribute '" + name + "': " + e.what());
        }
    }

    int64_t Attrs::Int(const std::string &name) const
    {
        const double value = Float(name);
        const double rounded = std::round(value);
        if (std::abs(value - rounded) > 1e-9)
        {
            std::ostringstream s;
            s << "attribute '" << name << "' must be a whole number, got " << value;
            throw GraphError(s.str());
        }
        return static_cast<int64_t>(rounded);
    }

    bool Attrs::Bool(const std::string &name) const
    {
        const AttrValue v = Raw(name);
        if (const auto *b = std::get_if<bool>(&v)) return *b;
        if (const auto *s = std::get_if<std::string>(&v))
        {
            if (*s == "true") return true;
            if (*s == "false") return false;
        }
        return Float(name) != 0.0;
    }

    std::string Attrs::Str(const std::string &name) const
    {
        return AttrText(Raw(name));
    }

    const TensorType &InferArgs::In(size_t i) const
    {
        if (i >= inputs.size() || !inputs[i])
        {
            throw GraphError("input " + std::to_string(i) + " is not connected");
        }
        return *inputs[i];
    }

    // ---- NodeSpec ------------------------------------------------------------

    const AttrSpec *NodeSpec::FindAttr(const std::string &name) const
    {
        for (const AttrSpec &a : attrs) if (a.name == name) return &a;
        return nullptr;
    }

    int NodeSpec::InputIndex(const std::string &name) const
    {
        for (size_t i = 0; i < inputs.size(); ++i) if (inputs[i].name == name) return static_cast<int>(i);
        return -1;
    }

    int NodeSpec::OutputIndex(const std::string &name) const
    {
        for (size_t i = 0; i < outputs.size(); ++i) if (outputs[i].name == name) return static_cast<int>(i);
        return -1;
    }

    int NodePorts::InputIndex(const std::string &name) const
    {
        for (size_t i = 0; i < inputs.size(); ++i) if (inputs[i].name == name) return static_cast<int>(i);
        return -1;
    }

    int NodePorts::OutputIndex(const std::string &name) const
    {
        for (size_t i = 0; i < outputs.size(); ++i) if (outputs[i].name == name) return static_cast<int>(i);
        return -1;
    }

    NodePorts NodeSpec::Ports(const Node &node, const Document &doc, const Registry &registry) const
    {
        if (dynamicPorts) return dynamicPorts(node, doc, registry);
        return NodePorts{inputs, outputs};
    }

    std::vector<int> OrderedIo(const Graph &graph, const Registry &registry, NodeRole role)
    {
        std::vector<std::pair<int64_t, int>> found;
        for (const Node &n : graph.nodes)
        {
            const SpecPtr spec = registry.Find(n.type);
            if (!spec || spec->role != role) continue;
            int64_t index = 0;
            if (const auto it = n.attrs.find("index"); it != n.attrs.end())
            {
                if (const auto *i = std::get_if<int64_t>(&it->second)) index = *i;
                else if (const auto *d = std::get_if<double>(&it->second)) index = static_cast<int64_t>(*d);
            }
            found.emplace_back(index, n.id);
        }
        std::sort(found.begin(), found.end());
        std::vector<int> ids;
        for (const auto &[index, id] : found) ids.push_back(id);
        return ids;
    }

    NodePorts SubgraphPorts(const Document &doc, const std::string &subgraph, const Registry &registry)
    {
        NodePorts ports;
        const auto it = doc.subgraphs.find(subgraph);
        if (it == doc.subgraphs.end()) return ports;
        const Graph &g = it->second;
        for (const int id : OrderedIo(g, registry, NodeRole::Input))
        {
            const Node &n = *g.Find(id);
            std::optional<DType> dtype = DType::F32;
            if (const auto a = n.attrs.find("dtype"); a != n.attrs.end() && AttrText(a->second) == "i64") dtype = DType::I64;
            if (n.type == "TokenIds") dtype = DType::I64;
            ports.inputs.push_back({n.name, dtype, false});
        }
        for (const int id : OrderedIo(g, registry, NodeRole::Output))
        {
            ports.outputs.push_back({g.Find(id)->name, std::nullopt, false});
        }
        return ports;
    }

    AttrMap NodeSpec::Defaults() const
    {
        AttrMap m;
        for (const AttrSpec &a : attrs) m[a.name] = a.def;
        return m;
    }

    // ---- Registry ------------------------------------------------------------

    void Registry::Register(NodeSpec spec)
    {
        std::unique_lock lock(m_mutex);
        const std::string type = spec.type;
        m_specs[type] = std::make_shared<const NodeSpec>(std::move(spec));
        ++m_version;
    }

    void Registry::Unregister(const std::string &type)
    {
        std::unique_lock lock(m_mutex);
        if (m_specs.erase(type)) ++m_version;
    }

    SpecPtr Registry::Find(const std::string &type) const
    {
        std::shared_lock lock(m_mutex);
        const auto it = m_specs.find(type);
        return it == m_specs.end() ? nullptr : it->second;
    }

    std::vector<SpecPtr> Registry::All() const
    {
        std::shared_lock lock(m_mutex);
        std::vector<SpecPtr> all;
        for (const auto &[k, v] : m_specs) all.push_back(v);
        return all;
    }

    uint64_t Registry::Version() const
    {
        std::shared_lock lock(m_mutex);
        return m_version;
    }

    // ---- Shapes --------------------------------------------------------------

    std::vector<Dim> ParseShape(const std::string &text, const Variables &vars)
    {
        std::vector<Dim> dims;
        std::stringstream stream(text);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            const size_t a = token.find_first_not_of(" \t");
            const size_t b = token.find_last_not_of(" \t");
            if (a == std::string::npos) throw GraphError("shape '" + text + "' has an empty dimension");
            token = token.substr(a, b - a + 1);

            // A bare name that is not a variable is a symbolic dimension (B, S, ...).
            std::string bare = token[0] == '$' ? token.substr(1) : token;
            if (IsIdentifier(bare) && !vars.contains(bare))
            {
                if (token[0] == '$') throw GraphError("shape '" + text + "': unknown variable '" + bare + "'");
                dims.push_back(Dim::Sym(bare));
                continue;
            }
            try
            {
                const double value = Evaluate(token, vars);
                dims.push_back(Dim::Known(static_cast<int64_t>(std::llround(value))));
            }
            catch (const std::exception &e)
            {
                throw GraphError("shape '" + text + "': " + e.what());
            }
        }
        return dims;
    }

    std::string ShapeError(const std::string &what, const TensorType &got)
    {
        return what + ", got " + got.Text();
    }
}
