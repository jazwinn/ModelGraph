// Python-defined nodes. Each .py file is executed in its own namespace and
// describes one node type with module-level names:
//
//   name = "Swish"                      # node type (required)
//   category, description               # palette grouping / tooltip
//   inputs = ["x"]; outputs = ["y"]     # or [{"name": "x", "dtype": "f32", "optional": True}]
//   attrs = {"beta": 1.0}               # bool/int/float/str; "$expr" strings are int expressions
//   def params(a): ...                  # optional {name: shape | {"shape":..., "init":..., "std":...}}
//   def shape(a, *inputs): ...          # optional output shape(s)
//   def forward(ctx, *inputs): ...      # required; returns a Tensor or a tuple of Tensors
//
// Tensors are InferenceEngine tensors (CPU or CUDA) exposed through the
// embedded `ie` module, so scripts compose the engine's kernels.

#include "Script/PyNodes.h"

#include <cmath>
#include <map>
#include <random>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Graph/Runtime.h"
#include "NeuralNet/Transformer/AttentionOps.h"

namespace py = pybind11;

namespace mg::script::detail
{
    thread_local tensor::Device g_device = tensor::Device::CPU;

    inline size_t Axis(int64_t axis, size_t rank)
    {
        const int64_t r = static_cast<int64_t>(rank);
        const int64_t a = axis < 0 ? axis + r : axis;
        if (a < 0 || a >= r) throw py::index_error("axis " + std::to_string(axis) + " out of range for rank " + std::to_string(rank));
        return static_cast<size_t>(a);
    }

    inline std::vector<size_t> Sizes(const py::args &args, size_t total)
    {
        // Accepts reshape(2, 3), reshape((2, 3)) or reshape([2, -1]).
        py::sequence seq = args;
        if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) seq = args[0].cast<py::sequence>();
        std::vector<size_t> dims;
        int infer = -1;
        size_t known = 1;
        for (const py::handle h : seq)
        {
            const int64_t d = h.cast<int64_t>();
            if (d == -1 && total)
            {
                if (infer >= 0) throw py::value_error("only one dimension may be -1");
                infer = static_cast<int>(dims.size());
                dims.push_back(1);
                continue;
            }
            if (d <= 0) throw py::value_error("shape entries must be positive");
            dims.push_back(static_cast<size_t>(d));
            known *= static_cast<size_t>(d);
        }
        if (infer >= 0)
        {
            if (total % known != 0) throw py::value_error("cannot infer the -1 dimension");
            dims[infer] = total / known;
        }
        return dims;
    }

    inline std::vector<size_t> Sizes(const py::object &shape)
    {
        std::vector<size_t> dims;
        for (const py::handle h : shape)
        {
            const int64_t d = h.cast<int64_t>();
            if (d <= 0) throw py::value_error("shape entries must be positive");
            dims.push_back(static_cast<size_t>(d));
        }
        return dims;
    }

    template <typename F>
    FTensor CpuMap(const FTensor &x, F f)
    {
        FTensor host = x.To(tensor::Device::CPU).Clone();
        Scalar *d = host.Data();
        for (size_t i = 0; i < host.Size(); ++i) d[i] = f(d[i]);
        return host.To(x.GetDevice());
    }

    inline tensor::Activation ActivationByName(const std::string &name)
    {
        if (name == "gelu") return tensor::Activation::GELU;
        if (name == "gelu_tanh") return tensor::Activation::GELUTanh;
        if (name == "silu") return tensor::Activation::SiLU;
        if (name == "relu") return tensor::Activation::ReLU;
        if (name == "sigmoid") return tensor::Activation::Sigmoid;
        if (name == "tanh") return tensor::Activation::Tanh;
        throw py::value_error("unknown activation '" + name + "'");
    }

    inline py::tuple ShapeOf(const FTensor &x)
    {
        py::tuple t(x.Rank());
        for (size_t i = 0; i < x.Rank(); ++i) t[i] = x.Shape()[i];
        return t;
    }
}

PYBIND11_EMBEDDED_MODULE(ie, m)
{
    using namespace mg;
    using namespace mg::script::detail;
    using T = FTensor;

    m.doc() = "InferenceEngine tensor operations for ModelGraph script nodes";

    py::class_<T>(m, "Tensor")
        .def_property_readonly("shape", &ShapeOf)
        .def_property_readonly("device", [](const T &x) { return x.IsCUDA() ? "cuda" : "cpu"; })
        .def("dim", [](const T &x, int64_t axis) { return x.Shape()[Axis(axis, x.Rank())]; })
        .def("rank", &T::Rank)
        .def("numel", &T::Size)
        .def("__len__", [](const T &x) { return x.Rank() ? x.Shape()[0] : 0; })
        .def("reshape", [](const T &x, py::args dims) { return x.Contiguous().Reshape(Sizes(dims, x.Size())); })
        .def("transpose", [](const T &x, int64_t a, int64_t b) { return x.Transpose(a, b).Contiguous(); }, py::arg("dim0") = -2, py::arg("dim1") = -1)
        .def("unsqueeze", [](const T &x, int64_t a) { return x.Unsqueeze(a < 0 ? Axis(a, x.Rank() + 1) : static_cast<size_t>(a)); })
        .def("squeeze", [](const T &x, int64_t a) { return x.Squeeze(Axis(a, x.Rank())); })
        .def("slice", [](const T &x, int64_t axis, int64_t start, int64_t count)
             { return x.Slice(Axis(axis, x.Rank()), static_cast<size_t>(start), static_cast<size_t>(count)).Contiguous(); })
        .def("contiguous", &T::Contiguous)
        .def("clone", &T::Clone)
        .def("mean", [](const T &x, int64_t axis, bool keepdim) { return tensor::Mean(x, Axis(axis, x.Rank()), keepdim); },
             py::arg("axis") = -1, py::arg("keepdim") = true)
        .def("to", [](const T &x, const std::string &device) { return x.To(device == "cuda" ? tensor::Device::CUDA : tensor::Device::CPU); })
        .def("numpy", [](const T &x)
             {
                 const T host = x.To(tensor::Device::CPU).Contiguous();
                 std::vector<py::ssize_t> shape(host.Shape().begin(), host.Shape().end());
                 py::array_t<Scalar> out(shape);
                 std::copy(host.Data(), host.Data() + host.Size(), out.mutable_data());
                 return out;
             })
        .def("tolist", [](const T &x) { return py::cast(x).attr("numpy")().attr("tolist")(); })
        .def("__add__", [](const T &a, const T &b) { return a + b; })
        .def("__add__", [](const T &a, double b) { return a + static_cast<Scalar>(b); })
        .def("__radd__", [](const T &a, double b) { return static_cast<Scalar>(b) + a; })
        .def("__sub__", [](const T &a, const T &b) { return a - b; })
        .def("__sub__", [](const T &a, double b) { return a - static_cast<Scalar>(b); })
        .def("__rsub__", [](const T &a, double b) { return static_cast<Scalar>(b) - a; })
        .def("__mul__", [](const T &a, const T &b) { return a * b; })
        .def("__mul__", [](const T &a, double b) { return a * static_cast<Scalar>(b); })
        .def("__rmul__", [](const T &a, double b) { return static_cast<Scalar>(b) * a; })
        .def("__truediv__", [](const T &a, const T &b) { return a / b; })
        .def("__truediv__", [](const T &a, double b) { return a / static_cast<Scalar>(b); })
        .def("__matmul__", [](const T &a, const T &b) { return tensor::MatMul(a, b); })
        .def("__neg__", [](const T &a) { return a * Scalar(-1); })
        .def("__repr__", [](const T &x)
             {
                 std::string s = "Tensor(";
                 for (size_t i = 0; i < x.Rank(); ++i) s += (i ? ", " : "") + std::to_string(x.Shape()[i]);
                 return s + (x.IsCUDA() ? "; cuda)" : "; cpu)");
             });

    m.def("matmul", [](const T &a, const T &b) { return tensor::MatMul(a, b); });
    m.def("linear", [](const T &x, const T &w, std::optional<T> b) { return tensor::Linear(x, w, b ? *b : T()); },
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("softmax", [](const T &x, int64_t axis) { return tensor::Softmax(x, axis); }, py::arg("x"), py::arg("axis") = -1);
    m.def("activate", [](const T &x, const std::string &name) { return tensor::Activate(x, ActivationByName(name)); });
    for (const char *name : {"gelu", "gelu_tanh", "silu", "relu", "sigmoid", "tanh"})
    {
        const tensor::Activation a = ActivationByName(name);
        m.def(name, [a](const T &x) { return tensor::Activate(x, a); });
    }
    m.def("sqrt", [](const T &x) { return tensor::Sqrt(x); });
    m.def("mean", [](const T &x, int64_t axis, bool keepdim) { return tensor::Mean(x, Axis(axis, x.Rank()), keepdim); },
          py::arg("x"), py::arg("axis") = -1, py::arg("keepdim") = true);
    m.def("sdpa", [](const T &q, const T &k, const T &v, bool causal, double scale)
          {
              return nn::transformer::SDPA(q.Contiguous(), k, v, tensor::Tensor<bool>(), causal,
                                           static_cast<Scalar>(scale != 0 ? scale : 1.0 / std::sqrt(static_cast<double>(q.Shape().back()))));
          },
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = true, py::arg("scale") = 0.0);
    m.def("split_heads", [](const T &x, size_t heads, size_t dim) { return nn::transformer::SplitHeads(x.Contiguous(), heads, dim); });
    m.def("merge_heads", [](const T &x, size_t width) { return nn::transformer::MergeHeads(x.Contiguous(), width); });
    m.def("zeros", [](const py::object &shape) { return T(Sizes(shape), Scalar(0), tensor::Device::CPU).To(g_device); });
    m.def("ones", [](const py::object &shape) { return T(Sizes(shape), Scalar(1), tensor::Device::CPU).To(g_device); });
    m.def("full", [](const py::object &shape, double v) { return T(Sizes(shape), static_cast<Scalar>(v), tensor::Device::CPU).To(g_device); });
    m.def("tensor", [](py::array_t<Scalar, py::array::c_style | py::array::forcecast> a)
          {
              std::vector<size_t> shape(a.shape(), a.shape() + a.ndim());
              if (shape.empty()) shape = {1};
              T t(shape);
              std::copy(a.data(), a.data() + a.size(), t.Data());
              return t.To(g_device);
          }, "Tensor from a numpy array or nested list (placed on the model's device)");

    // Element-wise maths without an engine kernel yet. These round-trip
    // through the CPU, so they are for prototyping rather than speed.
    m.def("exp", [](const T &x) { return CpuMap(x, [](Scalar v) { return std::exp(v); }); });
    m.def("log", [](const T &x) { return CpuMap(x, [](Scalar v) { return std::log(v); }); });
    m.def("abs", [](const T &x) { return CpuMap(x, [](Scalar v) { return std::abs(v); }); });
    m.def("pow", [](const T &x, double p) { return CpuMap(x, [p](Scalar v) { return static_cast<Scalar>(std::pow(v, p)); }); });
    m.def("clamp", [](const T &x, double lo, double hi)
          { return CpuMap(x, [lo, hi](Scalar v) { return static_cast<Scalar>(std::min(std::max(static_cast<double>(v), lo), hi)); }); });
    m.def("map", [](const T &x, const py::function &fn) { return CpuMap(x, [&fn](Scalar v) { return fn(v).cast<Scalar>(); }); },
          "apply a Python function to every element (slow)");
}

namespace mg::script
{
    struct ParamDecl
    {
        std::string name;
        std::vector<size_t> shape;
        std::string init = "normal";
        double std = 0.02;
        bool operator==(const ParamDecl &o) const { return name == o.name && shape == o.shape; }
    };

    struct PyParts
    {
        py::dict ns;
        py::object forward, shape, params;
    };

    struct ScriptState
    {
        NodeSpec spec;   // ports and attributes (no callbacks)
        std::filesystem::path file;
        std::unique_ptr<PyParts> py;

        ~ScriptState()
        {
            if (py)
            {
                py::gil_scoped_acquire gil;
                py.reset();
            }
        }
    };

    namespace
    {
        using detail::g_device;

        [[noreturn]] void Fail(const ScriptState &s, const std::string &message)
        {
            throw GraphError(s.file.filename().string() + ": " + message);
        }

        // Runs `f` with the GIL held, turning Python errors into GraphError.
        template <typename F>
        auto WithPython(const ScriptState &s, const char *what, F f)
        {
            py::gil_scoped_acquire gil;
            try
            {
                return f();
            }
            catch (py::error_already_set &e)
            {
                Fail(s, std::string(what) + ": " + e.what());
            }
            catch (const py::cast_error &e)
            {
                Fail(s, std::string(what) + ": " + e.what());
            }
        }

        py::object Namespace(const py::dict &values)
        {
            return py::module_::import("types").attr("SimpleNamespace")(**values);
        }

        // ---- reading the script ----------------------------------------------

        PortSpec ReadPort(const ScriptState &s, const py::handle &o)
        {
            PortSpec p;
            if (py::isinstance<py::str>(o))
            {
                p.name = o.cast<std::string>();
                return p;
            }
            if (!py::isinstance<py::dict>(o)) Fail(s, "inputs/outputs entries must be names or {'name': ..., 'dtype': ...}");
            const py::dict d = o.cast<py::dict>();
            if (!d.contains("name")) Fail(s, "every port needs a name");
            p.name = d["name"].cast<std::string>();
            const std::string dtype = d.contains("dtype") ? d["dtype"].cast<std::string>() : "f32";
            if (dtype == "any") p.dtype = std::nullopt;
            else p.dtype = ParseDType(dtype);
            p.optional = d.contains("optional") && d["optional"].cast<bool>();
            return p;
        }

        std::vector<PortSpec> ReadPorts(const ScriptState &s, const py::dict &ns, const char *key, const char *fallback)
        {
            std::vector<PortSpec> ports;
            if (!ns.contains(key))
            {
                ports.push_back({fallback, DType::F32, false});
                return ports;
            }
            for (const py::handle h : ns[key]) ports.push_back(ReadPort(s, h));
            return ports;
        }

        AttrSpec ReadAttr(const ScriptState &s, const std::string &name, const py::handle &o)
        {
            AttrSpec a;
            a.name = name;
            py::object def = py::reinterpret_borrow<py::object>(o);
            std::string kind;
            if (py::isinstance<py::dict>(o))
            {
                const py::dict d = o.cast<py::dict>();
                if (!d.contains("default")) Fail(s, "attribute '" + name + "' needs a 'default'");
                def = d["default"];
                if (d.contains("kind")) kind = d["kind"].cast<std::string>();
                if (d.contains("help")) a.help = d["help"].cast<std::string>();
                if (d.contains("options"))
                {
                    for (const py::handle h : d["options"]) a.options.push_back(h.cast<std::string>());
                    if (kind.empty()) kind = "enum";
                }
            }
            // bool before int: Python's bool is a subclass of int.
            if (py::isinstance<py::bool_>(def)) a.def = def.cast<bool>();
            else if (py::isinstance<py::int_>(def)) a.def = def.cast<int64_t>();
            else if (py::isinstance<py::float_>(def)) a.def = def.cast<double>();
            else if (py::isinstance<py::str>(def)) a.def = def.cast<std::string>();
            else Fail(s, "attribute '" + name + "' must default to a bool, int, float or str");

            if (kind.empty())
            {
                if (std::holds_alternative<bool>(a.def)) kind = "bool";
                else if (std::holds_alternative<int64_t>(a.def)) kind = "int";
                else if (std::holds_alternative<double>(a.def)) kind = "float";
                else kind = std::get<std::string>(a.def).starts_with("$") ? "int" : "string";
            }
            static const std::map<std::string, AttrKind> kinds = {
                {"bool", AttrKind::Bool}, {"int", AttrKind::Int}, {"float", AttrKind::Float}, {"string", AttrKind::String},
                {"enum", AttrKind::Enum}, {"shape", AttrKind::Shape}};
            const auto it = kinds.find(kind);
            if (it == kinds.end()) Fail(s, "attribute '" + name + "' has unknown kind '" + kind + "'");
            a.kind = it->second;
            return a;
        }

        // Requires the GIL.
        py::object AttrNamespace(const ScriptState &s, const AttrMap &values, const Variables &vars)
        {
            const Attrs attrs(s.spec, values, vars);
            py::dict d;
            for (const AttrSpec &a : s.spec.attrs)
            {
                switch (a.kind)
                {
                case AttrKind::Bool: d[py::str(a.name)] = attrs.Bool(a.name); break;
                case AttrKind::Int: d[py::str(a.name)] = attrs.Int(a.name); break;
                case AttrKind::Float: d[py::str(a.name)] = attrs.Float(a.name); break;
                default: d[py::str(a.name)] = attrs.Str(a.name); break;
                }
            }
            return Namespace(d);
        }

        py::tuple ShapeTuple(const TensorType &t)
        {
            py::tuple out(t.Rank());
            for (size_t i = 0; i < t.Rank(); ++i)
            {
                if (t.shape[i].IsKnown()) out[i] = py::int_(t.shape[i].value);
                else out[i] = py::str(t.shape[i].Text());
            }
            return out;
        }

        TensorType ReadShape(const ScriptState &s, const py::handle &o, DType dtype)
        {
            if (!py::isinstance<py::sequence>(o) || py::isinstance<py::str>(o)) Fail(s, "shape() must return shapes such as ('B', 'S', 64)");
            TensorType type{dtype, {}};
            for (const py::handle d : o)
            {
                if (py::isinstance<py::int_>(d) && !py::isinstance<py::bool_>(d)) type.shape.push_back(Dim::Known(d.cast<int64_t>()));
                else if (py::isinstance<py::str>(d)) type.shape.push_back(Dim::Sym(d.cast<std::string>()));
                else Fail(s, "shape entries must be ints or symbol names");
            }
            return type;
        }

        // Requires the GIL.
        std::vector<ParamDecl> EvalParams(const ScriptState &s, const AttrMap &values, const Variables &vars)
        {
            std::vector<ParamDecl> decls;
            if (!s.py->params) return decls;
            const py::object result = s.py->params(AttrNamespace(s, values, vars));
            if (!py::isinstance<py::dict>(result)) Fail(s, "params() must return a dict");
            for (const auto &[k, v] : result.cast<py::dict>())
            {
                ParamDecl d;
                d.name = k.cast<std::string>();
                if (py::isinstance<py::dict>(v))
                {
                    const py::dict spec = v.cast<py::dict>();
                    if (!spec.contains("shape")) Fail(s, "param '" + d.name + "' needs a 'shape'");
                    d.shape = detail::Sizes(spec["shape"]);
                    if (spec.contains("init")) d.init = spec["init"].cast<std::string>();
                    if (spec.contains("std")) d.std = spec["std"].cast<double>();
                }
                else
                {
                    d.shape = detail::Sizes(py::reinterpret_borrow<py::object>(v));
                }
                if (d.init != "normal" && d.init != "zeros" && d.init != "ones")
                {
                    Fail(s, "param '" + d.name + "': init must be 'normal', 'zeros' or 'ones'");
                }
                decls.push_back(std::move(d));
            }
            std::sort(decls.begin(), decls.end(), [](const ParamDecl &a, const ParamDecl &b) { return a.name < b.name; });
            return decls;
        }

        // ---- the kernel ------------------------------------------------------

        class ScriptKernel : public ModuleKernel
        {
        public:
            ScriptKernel(std::shared_ptr<ScriptSlot> slot, std::shared_ptr<ScriptState> built, AttrMap values, Variables vars,
                         std::vector<ParamDecl> decls, size_t outputs)
                : m_slot(std::move(slot)), m_built(std::move(built)), m_values(std::move(values)), m_vars(std::move(vars)),
                  m_decls(std::move(decls)), m_outputs(outputs)
            {
                std::mt19937_64 rng(std::hash<std::string>{}(m_built->spec.type));
                for (const ParamDecl &d : m_decls)
                {
                    auto p = std::make_shared<FTensor>(d.shape, d.init == "ones" ? Scalar(1) : Scalar(0));
                    if (d.init == "normal")
                    {
                        std::normal_distribution<float> normal(0.0f, static_cast<float>(d.std));
                        for (size_t i = 0; i < p->Size(); ++i) p->Data()[i] = normal(rng);
                    }
                    RegisterParameter(d.name, p);
                    m_params.emplace_back(d.name, p);
                }
            }

            std::vector<Value> Run(RunContext &ctx, const std::vector<Value> &in) override
            {
                const std::shared_ptr<ScriptState> state = Pick();
                return WithPython(*state, "forward()", [&]
                {
                    g_device = ctx.device;
                    py::dict params;
                    for (const auto &[name, p] : m_params) params[py::str(name)] = *p;
                    py::dict c;
                    c["attrs"] = AttrNamespace(*state, m_values, m_vars);
                    c["params"] = Namespace(params);
                    c["start_pos"] = ctx.startPos;
                    c["seq_len"] = ctx.seqLen;
                    c["device"] = ctx.device == tensor::Device::CUDA ? "cuda" : "cpu";

                    py::list args;
                    args.append(Namespace(c));
                    for (const Value &v : in)
                    {
                        if (const auto *f = std::get_if<FTensor>(&v)) args.append(py::cast(*f));
                        else if (std::holds_alternative<std::monostate>(v)) args.append(py::none());
                        else Fail(*state, "script inputs must be f32 tensors");
                    }
                    const py::object result = state->py->forward(*args);

                    std::vector<Value> out;
                    if (py::isinstance<FTensor>(result)) out.push_back(result.cast<FTensor>());
                    else if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result))
                    {
                        for (const py::handle h : result)
                        {
                            if (!py::isinstance<FTensor>(h)) Fail(*state, "forward() must return Tensors");
                            out.push_back(h.cast<FTensor>());
                        }
                    }
                    else Fail(*state, "forward() must return a Tensor or a tuple of Tensors");
                    if (out.size() != m_outputs)
                    {
                        Fail(*state, "forward() returned " + std::to_string(out.size()) + " tensors but the node has " +
                                         std::to_string(m_outputs) + " outputs");
                    }
                    return out;
                });
            }

        private:
            // Hot reload: use the newest script unless it changed the parameter
            // layout or output count, which needs a rebuild to take effect.
            std::shared_ptr<ScriptState> Pick()
            {
                std::shared_ptr<ScriptState> current = m_slot->Current();
                if (!current || current == m_built) return m_built;
                if (current != m_checked)
                {
                    m_checked = current;
                    try
                    {
                        py::gil_scoped_acquire gil;
                        m_checkedOk = current->spec.outputs.size() == m_outputs && EvalParams(*current, m_values, m_vars) == m_decls;
                    }
                    catch (const std::exception &)
                    {
                        m_checkedOk = false;
                    }
                }
                return m_checkedOk ? current : m_built;
            }

            std::shared_ptr<ScriptSlot> m_slot;
            std::shared_ptr<ScriptState> m_built;
            AttrMap m_values;
            Variables m_vars;
            std::vector<ParamDecl> m_decls;
            size_t m_outputs;
            std::vector<std::pair<std::string, std::shared_ptr<FTensor>>> m_params;
            std::shared_ptr<ScriptState> m_checked;
            bool m_checkedOk = false;
        };
    }

    void EnsurePython()
    {
        static std::once_flag once;
        std::call_once(once, []
        {
            if (Py_IsInitialized()) return;   // embedded in a host that already runs Python
            PyConfig config;
            PyConfig_InitPythonConfig(&config);
            config.install_signal_handlers = 0;   // leave Ctrl+C to the application
#ifdef MG_PYTHON_HOME
            // Find the standard library of the Python we were built against,
            // unless the user points PYTHONHOME somewhere else.
            size_t envLen = 0;
            getenv_s(&envLen, nullptr, 0, "PYTHONHOME");
            if (envLen == 0 &&std::filesystem::exists(std::filesystem::path(MG_PYTHON_HOME) / "Lib"))
            {
                PyConfig_SetBytesString(&config, &config.home, MG_PYTHON_HOME);
            }
#endif
            py::initialize_interpreter(&config, 0, nullptr, false);
            py::module_::import("ie");
            // Let any thread take the GIL from now on.
            PyEval_SaveThread();
        });
    }

    std::shared_ptr<ScriptState> LoadScript(const std::filesystem::path &file)
    {
        EnsurePython();
        auto s = std::make_shared<ScriptState>();
        s->file = file;

        WithPython(*s, "load", [&]
        {
            s->py = std::make_unique<PyParts>();
            py::dict &ns = s->py->ns;
            ns["__builtins__"] = py::module_::import("builtins");
            ns["__file__"] = file.string();
            ns["__name__"] = "mg_script_" + file.stem().string();
            ns["ie"] = py::module_::import("ie");

            const py::object io = py::module_::import("io");
            const py::object handle = io.attr("open")(file.string(), "r", py::arg("encoding") = "utf-8");
            const py::object source = handle.attr("read")();
            handle.attr("close")();
            const py::object code = py::module_::import("builtins").attr("compile")(source, file.string(), "exec");
            py::module_::import("builtins").attr("exec")(code, ns);

            NodeSpec &n = s->spec;
            if (!ns.contains("name")) Fail(*s, "missing `name = \"MyNode\"`");
            n.type = ns["name"].cast<std::string>();
            n.category = ns.contains("category") ? ns["category"].cast<std::string>() : "Custom";
            n.description = ns.contains("description") ? ns["description"].cast<std::string>() : "";
            n.inputs = ReadPorts(*s, ns, "inputs", "x");
            n.outputs = ReadPorts(*s, ns, "outputs", "y");
            n.source = file.string();
            if (ns.contains("attrs"))
            {
                if (!py::isinstance<py::dict>(ns["attrs"])) Fail(*s, "`attrs` must be a dict");
                for (const auto &[k, v] : ns["attrs"].cast<py::dict>()) n.attrs.push_back(ReadAttr(*s, k.cast<std::string>(), v));
            }

            auto function = [&](const char *key) -> py::object
            {
                if (!ns.contains(key)) return py::object();
                py::object f = ns[key];
                if (!PyCallable_Check(f.ptr())) Fail(*s, std::string("`") + key + "` must be a function");
                return f;
            };
            s->py->forward = function("forward");
            if (!s->py->forward) Fail(*s, "missing `def forward(ctx, x): ...`");
            s->py->shape = function("shape");
            s->py->params = function("params");
        });
        return s;
    }

    std::string ScriptType(const ScriptState &state) { return state.spec.type; }

    NodeSpec MakeSpec(const std::shared_ptr<ScriptSlot> &slot, const std::filesystem::path &file)
    {
        const std::shared_ptr<ScriptState> state = slot->Current();
        NodeSpec spec = state->spec;
        spec.source = file.string();
        if (spec.description.empty()) spec.description = "Python node from " + file.filename().string();

        spec.infer = [slot](const InferArgs &a) -> std::vector<TensorType>
        {
            const std::shared_ptr<ScriptState> s = slot->Current();
            for (size_t i = 0; i < s->spec.inputs.size(); ++i)
            {
                if (!s->spec.inputs[i].optional) (void)a.In(i);
            }
            const size_t nOut = s->spec.outputs.size();
            if (!s->py->shape)
            {
                if (a.inputs.empty() || !a.inputs[0]) Fail(*s, "nodes without inputs must define shape()");
                std::vector<TensorType> out(nOut, *a.inputs[0]);
                for (size_t i = 0; i < nOut; ++i) out[i].dtype = s->spec.outputs[i].dtype.value_or(a.inputs[0]->dtype);
                return out;
            }
            return WithPython(*s, "shape()", [&]
            {
                py::list args;
                args.append(AttrNamespace(*s, a.node.attrs, a.attrs.Vars()));
                for (const auto &t : a.inputs)
                {
                    if (t) args.append(ShapeTuple(*t));
                    else args.append(py::none());
                }
                const py::object r = s->py->shape(*args);
                auto dtype = [&](size_t i) { return s->spec.outputs[i].dtype.value_or(DType::F32); };
                std::vector<TensorType> out;
                if (nOut == 1)
                {
                    out.push_back(ReadShape(*s, r, dtype(0)));
                }
                else
                {
                    size_t i = 0;
                    for (const py::handle h : r)
                    {
                        if (i >= nOut) Fail(*s, "shape() returned more shapes than outputs");
                        out.push_back(ReadShape(*s, h, dtype(i++)));
                    }
                }
                return out;
            });
        };

        spec.params = [slot](const InferArgs &a) -> int64_t
        {
            const std::shared_ptr<ScriptState> s = slot->Current();
            return WithPython(*s, "params()", [&]
            {
                int64_t total = 0;
                for (const ParamDecl &d : EvalParams(*s, a.node.attrs, a.attrs.Vars()))
                {
                    int64_t n = 1;
                    for (size_t x : d.shape) n *= static_cast<int64_t>(x);
                    total += n;
                }
                return total;
            });
        };

        spec.build = [slot](BuildArgs &b) -> std::shared_ptr<NodeKernel>
        {
            const std::shared_ptr<ScriptState> s = slot->Current();
            std::vector<ParamDecl> decls = WithPython(*s, "params()", [&] { return EvalParams(*s, b.node.attrs, b.attrs.Vars()); });
            return std::make_shared<ScriptKernel>(slot, s, b.node.attrs, b.attrs.Vars(), std::move(decls), s->spec.outputs.size());
        };
        return spec;
    }
}
