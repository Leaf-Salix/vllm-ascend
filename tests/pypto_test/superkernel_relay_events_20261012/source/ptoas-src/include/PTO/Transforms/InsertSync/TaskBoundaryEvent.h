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
inline constexpr const char *kTaskBoundaryRelayIds = "pto.runtime_late_relay_event_ids";
inline constexpr const char *kTaskBoundaryRelayMarker = "PYPTO_TASK_BOUNDARY_RELAY_V1";
inline constexpr const char *kTaskBoundaryEventPipes[] = {"MTE2", "MTE1", "M", "FIX", "V"};
inline constexpr const char *kTaskBoundaryCubeRelayPairs[] = {
    "MTE2_TO_MTE1", "MTE1_TO_M", "M_TO_FIX", "FIX_TO_MTE3"};
inline constexpr const char *kTaskBoundaryVectorRelayPairs[] = {"MTE2_TO_V", "V_TO_MTE3"};

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

// Explicit synchronization can occupy every direct notification event. Track
// the ordinary load/compute chain as well, without borrowing an author's ID.
template <typename Pipe>
constexpr int taskBoundaryEventPairIndex(Pipe src, Pipe dst) {
  if (dst == Pipe::PIPE_MTE3) return taskBoundaryEventIndex(src);
  if (src == Pipe::PIPE_MTE2 && dst == Pipe::PIPE_MTE1) return 5;
  if (src == Pipe::PIPE_MTE1 && dst == Pipe::PIPE_M) return 6;
  if (src == Pipe::PIPE_M && dst == Pipe::PIPE_FIX) return 7;
  if (src == Pipe::PIPE_MTE2 && dst == Pipe::PIPE_V) return 8;
  return -1;
}

} // namespace pto
} // namespace mlir

#endif
