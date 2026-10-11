// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_TASK_BOUNDARY_EVENT_H
#define MLIR_DIALECT_PTO_TRANSFORMS_TASK_BOUNDARY_EVENT_H

namespace mlir {
namespace pto {

inline constexpr const char *kTaskBoundaryEventIds = "pto.runtime_late_event_ids";
inline constexpr const char *kTaskBoundaryEventMarker = "PYPTO_TASK_BOUNDARY_EVENTS_V1";
inline constexpr const char *kTaskBoundaryBlockingPoll = "pto.runtime_blocking_tail_poll";
inline constexpr const char *kTaskBoundaryBlockingMarker = "PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1";
inline constexpr const char *kTaskBoundaryEventPipes[] = {"MTE2", "MTE1", "M", "FIX", "V"};

template <typename Pipe>
constexpr int taskBoundaryEventIndex(Pipe src) {
  if (src == Pipe::PIPE_MTE2) return 0;
  if (src == Pipe::PIPE_MTE1) return 1;
  if (src == Pipe::PIPE_M) return 2;
  if (src == Pipe::PIPE_FIX) return 3;
  if (src == Pipe::PIPE_V) return 4;
  return -1;
}

// Used with both the IR PIPE enum and InsertSync's PipelineType. Keep manual
// event validation and automatic allocation on the same notification contract.
template <typename Pipe>
constexpr bool usesTaskBoundaryEvent7(Pipe src, Pipe dst) {
  return dst == Pipe::PIPE_MTE3 && taskBoundaryEventIndex(src) >= 0;
}

} // namespace pto
} // namespace mlir

#endif
