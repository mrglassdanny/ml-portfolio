#pragma once

#define ZERO_CORE_THROW_ERROR(zero_core_err_msg) \
    printf("%s", zero_core_err_msg);             \
    exit(1)