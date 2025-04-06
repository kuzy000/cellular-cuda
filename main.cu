#include "cellular.cu"
#include "app.cu"

#include <cstdlib>

int main(void)
{
    App app;
    
    if (!app.init()) {
        return EXIT_FAILURE;
    }
    
    if (!app.loop()) {
        return EXIT_FAILURE;
    }
    
    if (!app.term()) {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

