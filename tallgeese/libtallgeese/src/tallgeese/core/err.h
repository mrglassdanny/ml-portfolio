#pragma once

#define TALLGEESE_CORE_THROW_ERROR(tallgeese_core_err_msg) \
    printf("%s", tallgeese_core_err_msg);                  \
    exit(1)