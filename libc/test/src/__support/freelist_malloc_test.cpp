//===-- Unittests for freelist_malloc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/stdlib/aligned_alloc.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::freelist_heap;
using LIBC_NAMESPACE::FreeListHeap;
using LIBC_NAMESPACE::FreeListHeapBuffer;

TEST(LlvmLibcFreeListMalloc, Malloc) {
  void *ptr1 = LIBC_NAMESPACE::malloc(256);
  auto *block = Block::from_usable_space(ptr1);
  EXPECT_GE(block->inner_size(), size_t{256});

  LIBC_NAMESPACE::free(ptr1);
  ASSERT_NE(block->next(), static_cast<Block *>(nullptr));
  ASSERT_EQ(block->next()->next(), static_cast<Block *>(nullptr));
  size_t heap_size = block->inner_size();

  void *ptr2 = LIBC_NAMESPACE::calloc(4, 64);
  ASSERT_EQ(ptr2, ptr1);
  EXPECT_GE(block->inner_size(), size_t{4 * 64});

  for (size_t i = 0; i < 4 * 64; ++i)
    EXPECT_EQ(reinterpret_cast<uint8_t *>(ptr2)[i], uint8_t(0));

  LIBC_NAMESPACE::free(ptr2);
  EXPECT_EQ(block->inner_size(), heap_size);

  void *ptr3 = LIBC_NAMESPACE::aligned_alloc(256, 256);
  EXPECT_NE(ptr3, static_cast<void *>(nullptr));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 256, size_t(0));
  auto *aligned_block = reinterpret_cast<Block *>(ptr3);
  EXPECT_GE(aligned_block->inner_size(), size_t{256});

  LIBC_NAMESPACE::free(ptr3);
  EXPECT_EQ(block->inner_size(), heap_size);
}
