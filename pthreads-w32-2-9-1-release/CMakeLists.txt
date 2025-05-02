project(pthread LANGUAGES C)

add_compile_definitions(
        HAVE_STRUCT_TIMESPEC
        PTW32_STATIC_LIB
)

add_library(pthread STATIC
        config.h
        context.h
        implement.h
        need_errno.h
        pthread.h
        sched.h
        semaphore.h
        private.c
        attr.c
        barrier.c
        cancel.c
        cleanup.c
        condvar.c
        create.c
        autostatic.c
        errno.c
        exit.c
        fork.c
        global.c
        misc.c
        mutex.c
        nonportable.c
        rwlock.c
        sched.c
        semaphore.c
        signal.c
        spin.c
        sync.c
        tsd.c
)

target_include_directories(pthread INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})