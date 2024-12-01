//===-- Strftime related internals -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_TIME_DEF_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_TIME_DEF_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string_view.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

static constexpr int NUM_DAYS = 7;
static constexpr int NUM_MONTHS = 12;
static constexpr int YEAR_BASE = 1900;

/* The number of days from the first day of the first ISO week of this
   year to the year day YDAY with week day WDAY.  ISO weeks start on
   Monday; the first ISO week has the year's first Thursday.  YDAY may
   be as small as YDAY_MINIMUM.  */
static constexpr int ISO_WEEK_START_WDAY = 1; /* Monday */
static constexpr int ISO_WEEK1_WDAY = 4;      /* Thursday */
static constexpr int YDAY_MINIMUM = -366;

static constexpr cpp::array<cpp::string_view, NUM_DAYS> day_names = {
    "Sunday",   "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"};

static constexpr cpp::array<cpp::string_view, NUM_MONTHS> month_names = {
    "January", "February", "March",     "April",   "May",      "June",
    "July",    "August",   "September", "October", "November", "December"};

static constexpr cpp::array<cpp::string_view, NUM_DAYS> abbreviated_day_names =
    {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

static constexpr cpp::array<cpp::string_view, NUM_MONTHS>
    abbreviated_month_names = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

static constexpr cpp::string_view out_of_bound_str =
    "?"; // From glibc output ? when days out of range

LIBC_INLINE cpp::string_view safe_day_name(int day) {
  return (day < 0 || day > 6) ? out_of_bound_str : day_names[day];
}

LIBC_INLINE cpp::string_view safe_abbreviated_day_name(int day) {
  return (day < 0 || day > 6) ? out_of_bound_str : abbreviated_day_names[day];
}

LIBC_INLINE cpp::string_view safe_month_name(int month) {
  return (month < 0 || month > 11) ? out_of_bound_str : month_names[month];
}

LIBC_INLINE cpp::string_view safe_abbreviated_month_name(int month) {
  return (month < 0 || month > 11) ? out_of_bound_str
                                   : abbreviated_month_names[month];
}

static constexpr cpp::string_view default_timezone_name = "UTC";

// TODO
static constexpr cpp::string_view default_timezone_offset = "+0000";

static constexpr cpp::string_view default_PM_str = "PM";

static constexpr cpp::string_view default_AM_str = "AM";

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif
