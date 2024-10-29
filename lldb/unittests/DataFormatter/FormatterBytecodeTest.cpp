#include "DataFormatters/FormatterBytecode.h"
#include "lldb/Utility/StreamString.h"

#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using namespace FormatterBytecode;
using llvm::StringRef;

namespace {
class FormatterBytecodeTest : public ::testing::Test {};

bool Interpret(std::vector<uint8_t> code, DataStack &data) {
  auto buf = StringRef(reinterpret_cast<const char *>(code.data()), code.size());
  std::vector<ControlStackElement> control({buf});
  if (auto error = Interpret(control, data, sel_summary)) {
    llvm::errs() << llvm::toString(std::move(error)) <<"\n";
    return false;
  }
  return true;
}

} // namespace

TEST_F(FormatterBytecodeTest, Basic) {
  {
    DataStack data;
    ASSERT_TRUE(Interpret({op_lit_uint, 23, op_dup, op_plus}, data));
    ASSERT_EQ(data.Pop<uint64_t>(), (uint64_t)46);
  }
  {
    DataStack data;
    ASSERT_FALSE(Interpret({op_lit_uint, 23, op_lit_uint, 0, op_div}, data));
  }
}
