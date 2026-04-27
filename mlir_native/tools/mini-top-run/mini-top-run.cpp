#include "mlir_start/Dialect/MiniTop/IR/MiniTopOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::mlir_start::mini_top;

namespace {

enum class DType { F32, I8, I32, I64 };

struct Tensor {
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  std::vector<float> f32;
  std::vector<int8_t> i8;
  std::vector<int32_t> i32;
  std::vector<int64_t> i64;

  int64_t numel() const {
    if (shape.empty())
      return 1;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                           std::multiplies<int64_t>());
  }
};

static llvm::cl::opt<std::string>
    inputMlir("mlir", llvm::cl::desc("Input mini_top MLIR file"),
              llvm::cl::Required);
static llvm::cl::opt<std::string>
    inputNpy("input-npy", llvm::cl::desc("Single NCHW .npy input tensor"));
static llvm::cl::opt<std::string>
    inputList("input-list", llvm::cl::desc("Text file with one .npy input per line"));
static llvm::cl::opt<std::string>
    weightsDir("weights-dir", llvm::cl::desc("Directory containing one .npy per weight"));
static llvm::cl::opt<std::string>
    outputDir("output-dir", llvm::cl::init("mini_top_run_outputs"),
              llvm::cl::desc("Directory for .npy outputs or calibration JSON"));
static llvm::cl::opt<std::string>
    mode("mode", llvm::cl::init("run"),
         llvm::cl::desc("run, calibrate, or compare"));
static llvm::cl::opt<std::string>
    backend("backend", llvm::cl::init("cpu"),
            llvm::cl::desc("cpu or cuda-fused"));
static llvm::cl::opt<std::string>
    aliases("output-aliases", llvm::cl::init("350,498,646"),
            llvm::cl::desc("Comma-separated names for returned outputs"));

static int64_t product(ArrayRef<int64_t> dims) {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

static std::vector<std::string> splitCSV(StringRef raw) {
  std::vector<std::string> out;
  SmallVector<StringRef> parts;
  raw.split(parts, ',');
  for (StringRef part : parts) {
    part = part.trim();
    if (!part.empty())
      out.push_back(part.str());
  }
  return out;
}

static std::string sanitize(StringRef key) {
  std::string out;
  for (char c : key)
    out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
                   c == '-' || c == '.')
                      ? c
                      : '_');
  return out.empty() ? "tensor" : out;
}

static std::optional<std::vector<int64_t>> rankedShape(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked || !ranked.hasStaticShape())
    return std::nullopt;
  return std::vector<int64_t>(ranked.getShape().begin(), ranked.getShape().end());
}

static std::vector<int64_t> arrayAttr(Operation *op, StringRef name,
                                      ArrayRef<int64_t> fallback) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return std::vector<int64_t>(fallback.begin(), fallback.end());
  std::vector<int64_t> values;
  for (Attribute item : attr)
    values.push_back(cast<IntegerAttr>(item).getInt());
  return values;
}

static int64_t i64Attr(Operation *op, StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

static double f64Attr(Operation *op, StringRef name, double fallback) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return attr.getValueAsDouble();
  return fallback;
}

static Tensor makeF32(std::vector<int64_t> shape) {
  Tensor t;
  t.dtype = DType::F32;
  t.shape = std::move(shape);
  t.f32.assign(t.numel(), 0.0f);
  return t;
}

static float tensorGetF32(const Tensor &t, int64_t index) {
  switch (t.dtype) {
  case DType::F32:
    return t.f32[index];
  case DType::I8:
    return static_cast<float>(t.i8[index]);
  case DType::I32:
    return static_cast<float>(t.i32[index]);
  case DType::I64:
    return static_cast<float>(t.i64[index]);
  }
  return 0.0f;
}

static std::string readHeader(std::ifstream &is, uint16_t major) {
  if (major == 1) {
    uint16_t len = 0;
    is.read(reinterpret_cast<char *>(&len), sizeof(len));
    std::string header(len, '\0');
    is.read(header.data(), len);
    return header;
  }
  uint32_t len = 0;
  is.read(reinterpret_cast<char *>(&len), sizeof(len));
  std::string header(len, '\0');
  is.read(header.data(), len);
  return header;
}

static std::vector<int64_t> parseNpyShape(StringRef header) {
  size_t open = header.find('(');
  size_t close = header.find(')', open);
  if (open == StringRef::npos || close == StringRef::npos)
    llvm::report_fatal_error("Invalid npy shape header");
  std::vector<int64_t> shape;
  SmallVector<StringRef> parts;
  header.substr(open + 1, close - open - 1).split(parts, ',');
  for (StringRef part : parts) {
    part = part.trim();
    if (!part.empty())
      shape.push_back(std::stoll(part.str()));
  }
  return shape;
}

static Tensor loadNpy(StringRef path) {
  std::ifstream is(path.str(), std::ios::binary);
  if (!is)
    llvm::report_fatal_error("Failed to open npy file: " + path);
  char magic[6];
  is.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY")
    llvm::report_fatal_error("Invalid npy file: " + path);
  uint8_t major = 0, minor = 0;
  is.read(reinterpret_cast<char *>(&major), 1);
  is.read(reinterpret_cast<char *>(&minor), 1);
  std::string header = readHeader(is, major);
  Tensor t;
  t.shape = parseNpyShape(header);
  int64_t n = product(t.shape);
  if (header.find("'descr': '<f4'") != std::string::npos ||
      header.find("\"descr\": \"<f4\"") != std::string::npos) {
    t.dtype = DType::F32;
    t.f32.resize(n);
    is.read(reinterpret_cast<char *>(t.f32.data()), n * sizeof(float));
  } else if (header.find("'descr': '|i1'") != std::string::npos ||
             header.find("'descr': '<i1'") != std::string::npos) {
    t.dtype = DType::I8;
    t.i8.resize(n);
    is.read(reinterpret_cast<char *>(t.i8.data()), n * sizeof(int8_t));
  } else if (header.find("'descr': '<i4'") != std::string::npos) {
    t.dtype = DType::I32;
    t.i32.resize(n);
    is.read(reinterpret_cast<char *>(t.i32.data()), n * sizeof(int32_t));
  } else if (header.find("'descr': '<i8'") != std::string::npos) {
    t.dtype = DType::I64;
    t.i64.resize(n);
    is.read(reinterpret_cast<char *>(t.i64.data()), n * sizeof(int64_t));
  } else {
    llvm::report_fatal_error("Unsupported npy dtype in: " + path);
  }
  return t;
}

static void saveNpyF32(StringRef path, const Tensor &tensor) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    llvm::report_fatal_error("Failed to write npy file: " + path);
  std::string shape = "(";
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i)
      shape += ", ";
    shape += std::to_string(tensor.shape[i]);
  }
  if (tensor.shape.size() == 1)
    shape += ",";
  shape += ")";
  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
  size_t pad = 16 - ((10 + header.size() + 1) % 16);
  header.append(pad, ' ');
  header.push_back('\n');
  os.write("\x93NUMPY", 6);
  uint8_t major = 1, minor = 0;
  os.write(reinterpret_cast<char *>(&major), 1);
  os.write(reinterpret_cast<char *>(&minor), 1);
  uint16_t len = static_cast<uint16_t>(header.size());
  os.write(reinterpret_cast<char *>(&len), sizeof(len));
  os << header;
  std::vector<float> data(tensor.numel());
  for (int64_t i = 0; i < tensor.numel(); ++i)
    data[i] = tensorGetF32(tensor, i);
  os.write(reinterpret_cast<const char *>(data.data()),
           data.size() * sizeof(float));
}

static Tensor quantize(const Tensor &input, double scale) {
  Tensor out;
  out.dtype = DType::I8;
  out.shape = input.shape;
  out.i8.resize(input.numel());
  if (scale <= 1e-12)
    scale = 1.0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    float q = std::round(tensorGetF32(input, i) / static_cast<float>(scale));
    q = std::max(-127.0f, std::min(127.0f, q));
    out.i8[i] = static_cast<int8_t>(q);
  }
  return out;
}

static Tensor dequantize(const Tensor &input, double scale) {
  Tensor out = makeF32(input.shape);
  for (int64_t i = 0; i < input.numel(); ++i)
    out.f32[i] = tensorGetF32(input, i) * static_cast<float>(scale);
  return out;
}

static Tensor silu(const Tensor &input) {
  Tensor out = makeF32(input.shape);
  for (int64_t i = 0; i < input.numel(); ++i) {
    float x = tensorGetF32(input, i);
    out.f32[i] = x / (1.0f + std::exp(-x));
  }
  return out;
}

static Tensor binary(const Tensor &lhs, const Tensor &rhs, bool add) {
  Tensor out = makeF32(lhs.shape);
  int64_t rhsN = rhs.numel();
  for (int64_t i = 0; i < out.numel(); ++i) {
    float a = tensorGetF32(lhs, i);
    float b = tensorGetF32(rhs, rhsN == 1 ? 0 : i);
    out.f32[i] = add ? a + b : a * b;
  }
  return out;
}

static Tensor conv2d(const Tensor &input, const Tensor &weight,
                     const Tensor &bias, Operation *op) {
  auto outShape = *rankedShape(op->getResult(0).getType());
  Tensor out = makeF32(outShape);
  auto strides = arrayAttr(op, "strides", {1, 1});
  auto pads = arrayAttr(op, "pads", {0, 0, 0, 0});
  auto dilations = arrayAttr(op, "dilations", {1, 1});
  int64_t group = i64Attr(op, "group", 1);
  int64_t n = outShape[0], oc = outShape[1], oh = outShape[2], ow = outShape[3];
  int64_t ic = input.shape[1], ih = input.shape[2], iw = input.shape[3];
  int64_t kh = weight.shape[2], kw = weight.shape[3];
  int64_t ocPerGroup = oc / group;
  int64_t icPerGroup = ic / group;
  for (int64_t b = 0; b < n; ++b)
    for (int64_t o = 0; o < oc; ++o)
      for (int64_t y = 0; y < oh; ++y)
        for (int64_t x = 0; x < ow; ++x) {
          float acc = bias.numel() ? tensorGetF32(bias, o) : 0.0f;
          int64_t g = o / ocPerGroup;
          for (int64_t c = 0; c < icPerGroup; ++c)
            for (int64_t r = 0; r < kh; ++r)
              for (int64_t s = 0; s < kw; ++s) {
                int64_t inY = y * strides[0] + r * dilations[0] - pads[0];
                int64_t inX = x * strides[1] + s * dilations[1] - pads[1];
                if (inY < 0 || inY >= ih || inX < 0 || inX >= iw)
                  continue;
                int64_t inC = g * icPerGroup + c;
                int64_t inIdx = ((b * ic + inC) * ih + inY) * iw + inX;
                int64_t wtIdx = ((o * icPerGroup + c) * kh + r) * kw + s;
                acc += tensorGetF32(input, inIdx) * tensorGetF32(weight, wtIdx);
              }
          out.f32[((b * oc + o) * oh + y) * ow + x] = acc;
        }
  return out;
}

static Tensor maxpool(const Tensor &input, Operation *op) {
  auto outShape = *rankedShape(op->getResult(0).getType());
  Tensor out = makeF32(outShape);
  auto kernel = arrayAttr(op, "kernel_shape", {1, 1});
  auto strides = arrayAttr(op, "strides", {1, 1});
  auto pads = arrayAttr(op, "pads", {0, 0, 0, 0});
  int64_t n = outShape[0], c = outShape[1], oh = outShape[2], ow = outShape[3];
  int64_t ih = input.shape[2], iw = input.shape[3];
  for (int64_t b = 0; b < n; ++b)
    for (int64_t ch = 0; ch < c; ++ch)
      for (int64_t y = 0; y < oh; ++y)
        for (int64_t x = 0; x < ow; ++x) {
          float best = -std::numeric_limits<float>::infinity();
          for (int64_t r = 0; r < kernel[0]; ++r)
            for (int64_t s = 0; s < kernel[1]; ++s) {
              int64_t inY = y * strides[0] + r - pads[0];
              int64_t inX = x * strides[1] + s - pads[1];
              if (inY < 0 || inY >= ih || inX < 0 || inX >= iw)
                continue;
              best = std::max(best, tensorGetF32(input, ((b * c + ch) * ih + inY) * iw + inX));
            }
          out.f32[((b * c + ch) * oh + y) * ow + x] = best;
        }
  return out;
}

static Tensor concat(ArrayRef<Tensor> inputs, int64_t axis) {
  std::vector<int64_t> shape = inputs.front().shape;
  shape[axis] = 0;
  for (const Tensor &t : inputs)
    shape[axis] += t.shape[axis];
  Tensor out = makeF32(shape);
  if (axis != 1)
    llvm::report_fatal_error("mini-top-run concat currently supports axis=1");
  int64_t n = shape[0], h = shape[2], w = shape[3], offset = 0;
  for (const Tensor &t : inputs) {
    int64_t c = t.shape[1];
    for (int64_t b = 0; b < n; ++b)
      for (int64_t ch = 0; ch < c; ++ch)
        for (int64_t y = 0; y < h; ++y)
          for (int64_t x = 0; x < w; ++x)
            out.f32[((b * shape[1] + offset + ch) * h + y) * w + x] =
                tensorGetF32(t, ((b * c + ch) * h + y) * w + x);
    offset += c;
  }
  return out;
}

static Tensor reshapeLike(const Tensor &input, Operation *op) {
  Tensor out = input;
  out.shape = *rankedShape(op->getResult(0).getType());
  return out;
}

static Tensor permute(const Tensor &input, Operation *op) {
  auto outShape = *rankedShape(op->getResult(0).getType());
  auto order = arrayAttr(op, "order", {});
  Tensor out = makeF32(outShape);
  std::vector<int64_t> inStride(input.shape.size(), 1), outStride(outShape.size(), 1);
  for (int i = static_cast<int>(input.shape.size()) - 2; i >= 0; --i)
    inStride[i] = inStride[i + 1] * input.shape[i + 1];
  for (int i = static_cast<int>(outShape.size()) - 2; i >= 0; --i)
    outStride[i] = outStride[i + 1] * outShape[i + 1];
  std::vector<int64_t> outIdx(outShape.size()), inIdx(input.shape.size());
  for (int64_t linear = 0; linear < out.numel(); ++linear) {
    int64_t rem = linear;
    for (size_t i = 0; i < outShape.size(); ++i) {
      outIdx[i] = rem / outStride[i];
      rem %= outStride[i];
    }
    for (size_t i = 0; i < order.size(); ++i)
      inIdx[order[i]] = outIdx[i];
    int64_t inLinear = 0;
    for (size_t i = 0; i < inIdx.size(); ++i)
      inLinear += inIdx[i] * inStride[i];
    out.f32[linear] = tensorGetF32(input, inLinear);
  }
  return out;
}

static Tensor interpNearest(const Tensor &input, Operation *op) {
  auto outShape = *rankedShape(op->getResult(0).getType());
  Tensor out = makeF32(outShape);
  int64_t n = outShape[0], c = outShape[1], oh = outShape[2], ow = outShape[3];
  int64_t ih = input.shape[2], iw = input.shape[3];
  for (int64_t b = 0; b < n; ++b)
    for (int64_t ch = 0; ch < c; ++ch)
      for (int64_t y = 0; y < oh; ++y)
        for (int64_t x = 0; x < ow; ++x) {
          int64_t inY = std::min<int64_t>(ih - 1, y * ih / oh);
          int64_t inX = std::min<int64_t>(iw - 1, x * iw / ow);
          out.f32[((b * c + ch) * oh + y) * ow + x] =
              tensorGetF32(input, ((b * c + ch) * ih + inY) * iw + inX);
        }
  return out;
}

struct Runner {
  ModuleOp module;
  llvm::DenseMap<Value, Tensor> env;
  std::map<std::string, double> absmax;

  void record(Operation *op, const Tensor &tensor) {
    std::string key;
    llvm::raw_string_ostream os(key);
    op->print(os, OpPrintingFlags().skipRegions().elideLargeElementsAttrs());
    double m = 0.0;
    for (int64_t i = 0; i < tensor.numel(); ++i)
      m = std::max(m, std::abs(static_cast<double>(tensorGetF32(tensor, i))));
    absmax[os.str()] = std::max(absmax[os.str()], m);
  }

  Tensor executeOp(Operation *op, ArrayRef<Tensor> args) {
    StringRef name = op->getName().getStringRef();
    if (name == WeightOp::getOperationName()) {
      auto key = op->getAttrOfType<StringAttr>("weight_key").getValue();
      SmallString<256> path(weightsDir);
      llvm::sys::path::append(path, key.str() + ".npy");
      return loadNpy(path);
    }
    if (name == ConvOp::getOperationName())
      return conv2d(args[0], args[1], args[2], op);
    if (name == ConvSiLUOp::getOperationName())
      return silu(conv2d(args[0], args[1], args[2], op));
    if (name == SigmoidOp::getOperationName()) {
      Tensor out = makeF32(args[0].shape);
      for (int64_t i = 0; i < out.numel(); ++i)
        out.f32[i] = 1.0f / (1.0f + std::exp(-tensorGetF32(args[0], i)));
      return out;
    }
    if (name == SiLUOp::getOperationName())
      return silu(args[0]);
    if (name == MulOp::getOperationName())
      return binary(args[0], args[1], false);
    if (name == AddOp::getOperationName())
      return binary(args[0], args[1], true);
    if (name == ConcatOp::getOperationName())
      return concat(args, i64Attr(op, "axis", 1));
    if (name == MaxPoolOp::getOperationName())
      return maxpool(args[0], op);
    if (name == InterpOp::getOperationName())
      return interpNearest(args[0], op);
    if (name == ReshapeOp::getOperationName())
      return reshapeLike(args[0], op);
    if (name == PermuteOp::getOperationName())
      return permute(args[0], op);
    if (name == QuantizeOp::getOperationName())
      return quantize(args[0], f64Attr(op, "scale", 1.0));
    if (name == DequantizeOp::getOperationName())
      return dequantize(args[0], f64Attr(op, "scale", 1.0));
    if (name == QConvOp::getOperationName()) {
      Tensor deq = dequantize(args[0], f64Attr(op, "input_scale", 1.0));
      Tensor conv = conv2d(deq, args[1], args[2], op);
      return quantize(conv, f64Attr(op, "output_scale", 1.0));
    }
    if (name == QConvSiLUOp::getOperationName()) {
      Tensor deq = dequantize(args[0], f64Attr(op, "input_scale", 1.0));
      Tensor conv = silu(conv2d(deq, args[1], args[2], op));
      return quantize(conv, f64Attr(op, "output_scale", 1.0));
    }
    llvm::report_fatal_error("Unsupported mini_top op in runner: " + name);
  }

  std::vector<Tensor> runOne(const Tensor &input) {
    env.clear();
    auto func = *module.getOps<func::FuncOp>().begin();
    Block &entry = func.getBody().front();
    env[entry.getArgument(0)] = input;
    for (Operation &op : entry.without_terminator()) {
      std::vector<Tensor> args;
      for (Value operand : op.getOperands())
        args.push_back(env.lookup(operand));
      Tensor result = executeOp(&op, args);
      if (op.getNumResults() == 1)
        env[op.getResult(0)] = result;
      record(&op, result);
    }
    auto ret = cast<func::ReturnOp>(entry.getTerminator());
    std::vector<Tensor> outputs;
    for (Value value : ret.getOperands())
      outputs.push_back(env.lookup(value));
    return outputs;
  }
};

static void writeCalibration(StringRef path, const std::map<std::string, double> &stats) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    llvm::report_fatal_error("Failed to write calibration table: " + path);
  os << "{\n  \"meta\": {\"format\": \"mini_top_absmax_v1\"},\n  \"stats\": {\n";
  bool first = true;
  for (const auto &[key, value] : stats) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    \"" << sanitize(key) << "\": {\"absmax\": " << value << "}";
  }
  os << "\n  }\n}\n";
}

static std::vector<std::string> readInputList(StringRef path) {
  std::ifstream is(path.str());
  if (!is)
    llvm::report_fatal_error("Failed to open input list: " + path);
  std::vector<std::string> paths;
  std::string line;
  while (std::getline(is, line)) {
    StringRef trimmed(line);
    trimmed = trimmed.trim();
    if (!trimmed.empty())
      paths.push_back(trimmed.str());
  }
  return paths;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "mini_top reference runner\n");
  if (backend != "cpu") {
    llvm::errs() << "Only --backend=cpu is executable in this build; CUDA fused "
                    "kernels are compiled separately.\n";
    return 2;
  }

  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<MiniTopDialect>();
  auto file = llvm::MemoryBuffer::getFileOrSTDIN(inputMlir);
  if (!file)
    llvm::report_fatal_error("Failed to open MLIR file");
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>((*file)->getBuffer(), &context);
  if (!module || failed(verify(*module)))
    llvm::report_fatal_error("Failed to parse/verify MLIR module");
  if (weightsDir.empty())
    llvm::report_fatal_error("--weights-dir is required");
  std::filesystem::create_directories(outputDir.getValue());

  Runner runner;
  runner.module = *module;
  std::vector<std::string> names = splitCSV(aliases);

  if (mode == "calibrate") {
    if (inputList.empty())
      llvm::report_fatal_error("--input-list is required for calibration");
    for (const std::string &path : readInputList(inputList))
      runner.runOne(loadNpy(path));
    SmallString<256> path(outputDir);
    llvm::sys::path::append(path, "calibration_table.json");
    writeCalibration(path, runner.absmax);
    llvm::outs() << "Wrote calibration table: " << path << "\n";
    return 0;
  }

  if (inputNpy.empty())
    llvm::report_fatal_error("--input-npy is required for run/compare");
  std::vector<Tensor> outputs = runner.runOne(loadNpy(inputNpy));
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string name = i < names.size() ? names[i] : ("out" + std::to_string(i));
    SmallString<256> path(outputDir);
    llvm::sys::path::append(path, sanitize(name) + ".npy");
    saveNpyF32(path, outputs[i]);
    llvm::outs() << "output[" << i << "] " << name << " -> " << path << "\n";
  }
  if (mode == "compare")
    llvm::outs() << "compare mode currently writes runner outputs for external comparison\n";
  return 0;
}
