#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "Tensor/Tensor.h"

namespace mg
{
    using Scalar = float;
    using FTensor = tensor::Tensor<Scalar>;
    using ITensor = tensor::Tensor<int64_t>;

    enum class DType { F32, I64 };

    const char *DTypeName(DType dtype);
    DType ParseDType(const std::string &text);

    // A dimension is either a known size or a named symbol such as "B" or "S"
    // that is only fixed at run time.
    struct Dim
    {
        int64_t value = -1;
        std::string symbol;

        static Dim Known(int64_t v) { return Dim{v, {}}; }
        static Dim Sym(std::string s) { return Dim{-1, std::move(s)}; }

        bool IsKnown() const { return symbol.empty() && value >= 0; }
        std::string Text() const { return IsKnown() ? std::to_string(value) : (symbol.empty() ? "?" : symbol); }
        bool operator==(const Dim &other) const { return value == other.value && symbol == other.symbol; }
    };

    struct TensorType
    {
        DType dtype = DType::F32;
        std::vector<Dim> shape;

        size_t Rank() const { return shape.size(); }
        std::string Text() const;
        bool operator==(const TensorType &other) const { return dtype == other.dtype && shape == other.shape; }
    };

    // Attribute values as stored in a graph file. A string in a numeric slot is
    // an expression over the document variables, e.g. "$d_model * 4".
    using AttrValue = std::variant<bool, int64_t, double, std::string>;
    using AttrMap = std::map<std::string, AttrValue>;

    std::string AttrText(const AttrValue &value);

    // A value flowing along a link at run time.
    using Value = std::variant<std::monostate, FTensor, ITensor>;

    const FTensor &AsFloat(const Value &value, const std::string &where);
    const ITensor &AsInt(const Value &value, const std::string &where);

    struct GraphError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
}
