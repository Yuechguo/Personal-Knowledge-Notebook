benchmark 复现情况：
```
#0  0x00007f297c33051d in syscall () from /lib64/libc.so.6
#1  0x00007f297dd9c009 in nsync::futex (val3=-1, uaddr2=0x0, timeout=<optimized out>, val=0, op=393, uaddr=0x7f0773269008) at external/nsync/platform/linux/src/nsync_semaphore_futex.c:108
#2  nsync::nsync_mu_semaphore_p_with_deadline (s=0x7f0773269008, abs_deadline=...) at external/nsync/platform/linux/src/nsync_semaphore_futex.c:108
#3  0x00007f297dd9a543 in nsync::nsync_cv_wait_with_deadline_generic (pcv=<optimized out>, pmu=<optimized out>, lock=lock@entry=0x7f297dd99f30 <nsync::void_mu_lock(void*)>, unlock=unlock@entry=0x7f297dd9a300 <nsync::void_mu_unlock(void*)>, abs_deadline=..., cancel_note=<optimized out>) at external/nsync/internal/cv.c:246
#4  0x00007f297dd9aa43 in nsync::nsync_cv_wait_with_deadline (pcv=<optimized out>, pmu=<optimized out>, abs_deadline=..., cancel_note=<optimized out>) at external/nsync/internal/cv.c:438
#5  0x00007f29888ceb7d in tensorflow::Notification::WaitForNotification (this=0x7f014792d360) at ./tensorflow/core/platform/default/notification.h:54
#6  tensorflow::DirectSession::WaitForNotification (this=this@entry=0x7f020f329e00, notification=notification@entry=0x7f014792d360, timeout_in_ms=<optimized out>) at tensorflow/core/common_runtime/direct_session.cc:3621
#7  0x00007f29888cec4b in tensorflow::DirectSession::WaitForNotification (this=this@entry=0x7f020f329e00, run_state=run_state@entry=0x7f014792d330, cm=cm@entry=0x7f014792d2a0, timeout_in_ms=<optimized out>) at tensorflow/core/common_runtime/direct_session.cc:3595
#8  0x00007f29888d9e07 in tensorflow::DirectSession::RunInternal (this=this@entry=0x7f020f329e00, step_id=step_id@entry=38068, query_priority=query_priority@entry=-1, run_options=..., call_frame=call_frame@entry=0x7f014792d850, executors_and_keys=0x7f01b2247e00, run_metadata=0x0, threadpool_options=..., blaze_stream_id=<optimized out>, cuda_graph_meta=0x0) at bazel-out/k8-opt/bin/tensorflow/core/protobuf/config.pb.h:9850
#9  0x00007f29888ed952 in tensorflow::DirectSession::RunCallable (this=0x7f020f329e00, handle=<optimized out>, feed_tensors=..., fetch_tensors=0x7f014792d9b0, run_metadata=0x0, threadpool_options=..., blaze_stream_id=10, before_padding=0, after_padding=0) at bazel-out/k8-opt/bin/tensorflow/core/protobuf/config.pb.h:11504
#10 0x00007f29888ca06b in tensorflow::DirectSession::RunCallable (this=<optimized out>, handle=<optimized out>, feed_tensors=..., fetch_tensors=<optimized out>, run_metadata=<optimized out>, blaze_stream_id=10, before_padding=0, after_padding=0) at tensorflow/core/common_runtime/direct_session.cc:3693
#11 0x00007f2984408234 in tensorflow::BlazeXlaPredictor::ComputeNoPadding (this=0x7f01b1ca3100, ctx=0x7f01d1e15670, inputs=...) at ./tensorflow/core/framework/op_kernel.h:786
#12 0x00007f298440e6c1 in tensorflow::BlazeXlaPredictor::Compute (this=0x7f01b1ca3100, ctx=0x7f01d1e15670) at tensorflow/core/kernels/blaze_xla_predictor.cc:553
#13 0x00000000004b7188 in tensorflow::BlazeXlaOp::Running(tensorflow::OpKernelContext*, tensorflow::BlazePredictor*, std::function<void ()> const&, unsigned long long) ()
#14 0x00007f297d94c191 in std::function<void ()>::operator()() const (this=<optimized out>) at /usr/lib/gcc/x86_64-redhat-linux/10/../../../../include/c++/10/bits/std_function.h:617
#15 tensorflow::thread::EigenEnvironment::ExecuteTask (t=..., this=0x7f01ab9b1668) at tensorflow/core/lib/core/threadpool.cc:84
#16 Eigen::ThreadPoolTempl<tensorflow::thread::EigenEnvironment>::WorkerLoop (this=<optimized out>, thread_id=<optimized out>) at external/eigen_archive/unsupported/Eigen/CXX11/src/ThreadPool/NonBlockingThreadPool.h:326
#17 0x00007f297d949a68 in std::function<void ()>::operator()() const (this=0x7f01b3953738) at /usr/lib/gcc/x86_64-redhat-linux/10/../../../../include/c++/10/bits/std_function.h:617
#18 tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}::operator()() const (__closure=0x7f01b3953730) at tensorflow/core/lib/core/threadpool.cc:61
#19 std::__invoke_impl<void, tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(std::__invoke_other, tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&) (__f=...) at /usr/lib/gcc/x86_64-redhat-linux/10/../../../../include/c++/10/bits/invoke.h:60
#20 std::__invoke_r<void, tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(void&&, (tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&)...) (__fn=...) at /usr/lib/gcc/x86_64-redhat-linux/10/../../../../include/c++/10/bits/invoke.h:110
```


可能的异常点：
```
0  0x00007f297c33051d in syscall () from /lib64/libc.so.6
#1  0x00007f297dd9c009 in nsync::futex (val3=-1, uaddr2=0x0, timeout=<optimized out>, val=0, op=393, uaddr=0x7f018b2d2008) at external/nsync/platform/linux/src/nsync_semaphore_futex.c:108
#2  nsync::nsync_mu_semaphore_p_with_deadline (s=0x7f018b2d2008, abs_deadline=...) at external/nsync/platform/linux/src/nsync_semaphore_futex.c:108
#3  0x00007f297dd9a543 in nsync::nsync_cv_wait_with_deadline_generic (pcv=<optimized out>, pmu=<optimized out>, lock=lock@entry=0x7f297dd99f30 <nsync::void_mu_lock(void*)>, unlock=unlock@entry=0x7f297dd9a300 <nsync::void_mu_unlock(void*)>, abs_deadline=..., cancel_note=<optimized out>) at external/nsync/internal/cv.c:246
#4  0x00007f297dd9aa43 in nsync::nsync_cv_wait_with_deadline (pcv=<optimized out>, pmu=<optimized out>, abs_deadline=..., cancel_note=<optimized out>) at external/nsync/internal/cv.c:438
#5  0x00007f29888ceb7d in tensorflow::Notification::WaitForNotification (this=0x7f01fcff3fd0) at ./tensorflow/core/platform/default/notification.h:54
#6  tensorflow::DirectSession::WaitForNotification (this=this@entry=0x7f020f328000, notification=notification@entry=0x7f01fcff3fd0, timeout_in_ms=<optimized out>) at tensorflow/core/common_runtime/direct_session.cc:3621
#7  0x00007f29888cec4b in tensorflow::DirectSession::WaitForNotification (this=this@entry=0x7f020f328000, run_state=run_state@entry=0x7f01fcff3fa0, cm=cm@entry=0x7f01fcff3f10, timeout_in_ms=<optimized out>) at tensorflow/core/common_runtime/direct_session.cc:3595
#8  0x00007f29888d9e07 in tensorflow::DirectSession::RunInternal (this=this@entry=0x7f020f328000, step_id=step_id@entry=38062, query_priority=query_priority@entry=671416406, run_options=..., call_frame=call_frame@entry=0x7f01fcff45c0, executors_and_keys=0x7f020f318300, run_metadata=0x7f01fcff47e0, threadpool_options=..., blaze_stream_id=<optimized out>, cuda_graph_meta=0x0) at bazel-out/k8-opt/bin/tensorflow/core/protobuf/config.pb.h:9850
#9  0x00007f29888ec550 in tensorflow::DirectSession::Run (this=<optimized out>, run_options=..., inputs=..., output_names=std::vector of length 1, capacity 1 = {...}, target_nodes=..., outputs=0x7f01fcff47c0, run_metadata=0x7f01fcff47e0) at tensorflow/core/common_runtime/direct_session.cc:1900
#10 0x000000000047a218 in benchmark::Model::Warmup() () at /usr/include/c++/10/bits/exception.h:66
#11 0x000000000047b27b in benchmark::ModelReloader::CreateObject(bool, benchmark::Model const*) () at /usr/include/c++/10/bits/exception.h:66
#12 0x000000000047fdd5 in benchmark::DoubleBufferReloader<benchmark::Model>::Switch(bool, bool) () at /usr/include/c++/10/bits/exception.h:66
#13 0x00000000004763fa in benchmark::ModelSelector::Start() () at /usr/include/c++/10/bits/exception.h:66
#14 0x00007f297c6e8b74 in execute_native_thread_routine () from /lib64/libstdc++.so.6
#15 0x00007f29914e73f9 in start_thread () from /lib64/libpthread.so.0
#16 0x00007f297c335b13 in clone () from /lib64/libc.so.6
```

enable kfd debug thing：
`Debug for hang`
```
echo Y | sudo tee /sys/module/amdgpu/parameters/debug_evictions

echo "func restore_process_worker +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control

echo "func svm_range_restore_work +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control

echo "func kfd_ioctl_svm +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control

echo "func kgd_gfx_v9_hqd_reset +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control
```


`Gernal how to enable debug in `
**Enable dynamic debugging in kernel**  
 sudo sh -c "echo 'module amdgpu +plmf' > /sys/kernel/debug/dynamic_debug/control"  
  
**DEBUG KFD function:**  
echo "func event_interrupt_wq_v9 +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control   
  
**DEBUG KFD file:**  
echo "file kfd_interrupt.c +pfl" | sudo tee /sys/kernel/debug/dynamic_debug/control