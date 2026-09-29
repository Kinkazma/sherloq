#include <Python.h>
#include <mach-o/dyld.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    char executable[PATH_MAX], resolved[PATH_MAX], config[PATH_MAX];
    uint32_t length = sizeof(executable);
    if (_NSGetExecutablePath(executable, &length) || !realpath(executable, resolved)) return 1;
    char *slash = strrchr(resolved, '/');
    if (!slash) return 1;
    *slash = '\0';
    if (snprintf(config, sizeof(config), "%s/../Resources/install-root.txt", resolved) >= sizeof(config)) return 1;
    FILE *stream = fopen(config, "r");
    char root[PATH_MAX];
    if (!stream || !fgets(root, sizeof(root), stream)) {
        fprintf(stderr, "SHERLOQ: missing installation location. Rebuild the application.\n");
        if (stream) fclose(stream);
        return 1;
    }
    fclose(stream);
    root[strcspn(root, "\r\n")] = '\0';
    if (!root[0] || chdir(root)) return 1;
    unsetenv("PYTHONHOME"); unsetenv("PYTHONPATH");
    setenv("PYTHONNOUSERSITE", "1", 1);
    mkdir("logs", 0700);
    int fd = open("logs/runtime.log", O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
    char python[PATH_MAX], entry[PATH_MAX];
    if (snprintf(python, sizeof(python), "%s/venv/bin/python", root) >= sizeof(python) ||
        snprintf(entry, sizeof(entry), "%s/app_start.py", root) >= sizeof(entry)) return 1;
    char **args = calloc(argc + 3, sizeof(char *));
    if (!args) return 1;
    args[0] = python; args[1] = "-u"; args[2] = entry;
    for (int i = 1; i < argc; i++) args[i + 2] = argv[i];
    int status = Py_BytesMain(argc + 2, args);
    free(args);
    return status;
}
