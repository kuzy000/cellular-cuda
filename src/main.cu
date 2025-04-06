#include "app.cu"
#include "cellular.cu"

int main(void) {
  App app;

  if (!app.init()) {
    return 1;
  }

  if (!app.loop()) {
    return 1;
  }

  if (!app.term()) {
    return 1;
  }

  return 0;
}
