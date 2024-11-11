#ifndef SRC_WORKER_H_
#define SRC_WORKER_H_

#include <kizunapi.h>

namespace etjs {

// Shared data between main thread and worker.
template<typename R>
struct WorkerData {
  napi_env env = nullptr;
  napi_async_work work = nullptr;
  napi_deferred deffered = nullptr;

  std::function<R()> callback;
  std::unique_ptr<R> result;

  ~WorkerData() {
    if (deffered) {
      napi_reject_deferred(env, deffered,
                           ki::ToNodeValue(env, "Worker failed."));
    }
    if (work) {
      napi_delete_async_work(env, work);
    }
  }
};

// Do work in worker and return a Promise that resolves on finish.
template<typename R>
napi_value RunInWorker(napi_env env,
                       const char* name,
                       std::function<R()> callback) {
  std::unique_ptr<WorkerData<R>> data = std::make_unique<WorkerData<R>>();
  data->env = env;
  if (napi_create_async_work(
      env,
      nullptr,
      ki::ToNodeValue(env, name),
      [](napi_env env, void* hint) {
        auto* data = static_cast<WorkerData<R>*>(hint);
        data->result = std::make_unique<R>(data->callback());
      },
      [](napi_env env, napi_status status, void* hint) {
        auto* data = static_cast<WorkerData<R>*>(hint);
        // Resolve promise and release everything on complete.
        napi_resolve_deferred(env,
                              data->deffered,
                              ki::ToNodeValue(env, *data->result));
        data->deffered = nullptr;
        delete data;
      },
      data.get(),
      &data->work) != napi_ok) {
    ki::ThrowError(env, "Failed to create async work");
    return nullptr;
  }
  // Create the returned promise.
  napi_value result;
  if (napi_create_promise(env, &data->deffered, &result) != napi_ok) {
    ki::ThrowError(env, "Failed to create promise");
    return nullptr;
  }
  // Start the work.
  data->callback = std::move(callback);
  if (napi_queue_async_work(env, data->work) != napi_ok) {
    ki::ThrowError(env, "Failed to queue async work");
    return nullptr;
  }
  // Leak the data, which will be freed in complete handler.
  data.release();
  return result;
}

}  // namespace etjs

#endif  // SRC_WORKER_H_
