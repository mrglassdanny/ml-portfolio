#pragma once

#define EPYON_CORE_THROW_ERROR(epyon_core_err_msg) \
    printf("%s", epyon_core_err_msg);             \
    exit(1)