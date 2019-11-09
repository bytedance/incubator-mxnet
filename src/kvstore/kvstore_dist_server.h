/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <mxnet/c_api.h>
#include <mxnet/kvstore.h>
#include <ps/ps.h>
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include <fstream>
#include <chrono>
#include <atomic>
#include "../profiler/profiler.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"
#include "cpu_reducer.h"

namespace mxnet {
namespace kvstore {

// maintain same order in frontend.
enum class CommandType {
  kController, kSetMultiPrecision, kStopServer, kSyncMode,
  kSetGradientCompression, kSetProfilerParams
};

enum class RequestType {
  kDefaultPushPull, kRowSparsePushPull, kCompressedPushPull
};

struct DataHandleType {
  RequestType requestType;
  int dtype;
};

/*!
 * Uses Cantor pairing function to generate a unique number given two numbers.
 * This number can also be inverted to find the unique pair whose Cantor value is this number.
 * Ref: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
 * \param requestType RequestType
 * \param dtype integer
 * \return Cantor value of arguments
 */
static int GetCommandType(RequestType requestType, int d) {
  int m = static_cast<int>(requestType);
  return (((m + d) * (m + d + 1)) / 2) + d;
}

/*!
 * Unpairs Cantor value and finds the two integers used to pair.
 * Then returns DataHandleType object with those numbers.
 * \param cmd DataHandleCommand generated by GetCommandType function
 * \return DataHandleType
 */
static DataHandleType DepairDataHandleType(int cmd) {
  int w = std::floor((std::sqrt(8 * cmd + 1) - 1)/2);
  int t = ((w * w) + w) / 2;
  int y = cmd - t;
  int x = w - y;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  DataHandleType type;
  type.requestType = static_cast<RequestType>(x);
  type.dtype = y;
  return type;
}

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f();
        blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<char>(0);
    enable_pull_zero_copy_ = dmlc::GetEnv("ENABLE_PULL_ZERO_COPY", true);
    if (enable_pull_zero_copy_) {
      LOG(INFO) << "Enable zero copy of pull operations.";
    }

    log_key_info_ = dmlc::GetEnv("PS_KEY_LOG", false);
    if (log_key_info_) {
      LOG(INFO) << "Log key information at PS";
    }

    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
    update_buf_wait_ = dmlc::GetEnv("PS_ENABLE_GRADIENT_WAIT", false);

    sync_mode_ = !dmlc::GetEnv("BYTEPS_ENABLE_ASYNC", false);
    if (!sync_mode_) {
      LOG(INFO) << "BytePS server is enabled asynchronous training";
    }

    LOG(INFO) << "---------------- Add this line to verify it is working ----------------"; 
  }

  ~KVStoreDistServer() {
    profiler::Profiler::Get()->SetState(profiler::Profiler::ProfilerState(0));
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  struct UpdateBuf {
    std::vector<ps::KVMeta> request;
    NDArray merged;
    // temp_array is used to cast received values as float32 for computation if required
    NDArray temp_array;
  };

  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    CommandType recved_type = static_cast<CommandType>(recved.head);
    switch (recved_type) {
      case CommandType::kStopServer:
        exec_.Stop();
        break;
      case CommandType::kSyncMode:
        CHECK(0) << "kSyncMode is not available now";
        break;
      case CommandType::kSetGradientCompression:
        gradient_compression_->DecodeParams(recved.body);
        break;
      case CommandType::kSetProfilerParams:
        // last char is the type of profiler command
        ProcessServerProfilerCommands(static_cast<KVStoreServerProfilerCommand>
                                                  (recved.body.back() - '0'),
                                      recved.body);
        break;
      case CommandType::kSetMultiPrecision:
        // uses value 1 for message id from frontend
        CHECK(0) << "kSetMultiPrecision is not available now";
        break;
      case CommandType::kController:
        // this uses value 0 for message id from frontend
        // let the main thread to execute ctrl, which is necessary for python
        exec_.Exec([this, recved]() {
            CHECK(controller_);
            controller_(recved.head, recved.body);
          });
        break;
    }
    app->Response(recved);
  }

  /*
   * For keys already initialized, if necessary create stored_realt.
   * This will only be used if by some wrong usage of kvstore,
   * some keys are initialized before optimizer is set.
   */
  void CreateMultiPrecisionCopies() {
    for (auto const &stored_entry : store_) {
      const int key = stored_entry.first;
      const NDArray &stored = stored_entry.second;
      if (stored.dtype() != mshadow::kFloat32) {
        auto &stored_realt = store_realt_[key];
        if (stored.storage_type() == kRowSparseStorage) {
          stored_realt = NDArray(kRowSparseStorage, stored.shape(), stored.ctx(),
                                 true, mshadow::kFloat32);
        } else {
          stored_realt = NDArray(stored.shape(), stored.ctx(), false, mshadow::kFloat32);
        }

        auto &update = update_buf_[key];
        if (!update.merged.is_none()) {
          if (update.merged.storage_type() == kRowSparseStorage) {
            update.merged = NDArray(kRowSparseStorage, update.merged.shape(), update.merged.ctx(),
                                    true, mshadow::kFloat32);
          } else {
            update.merged = NDArray(update.merged.shape(), update.merged.ctx(), false,
                                    mshadow::kFloat32);
          }
        }
        CHECK(update.request.size() == 0)
          << ps::MyRank() << "Multiprecision mode can not be set while pushes are underway."
          << "Please set optimizer before pushing keys." << key << " " << update.request.size();

        CopyFromTo(stored, stored_realt);
      }
    }
    for (auto const &stored_realt_entry : store_realt_) {
      stored_realt_entry.second.WaitToRead();
    }
  }

  void ProcessServerProfilerCommands(KVStoreServerProfilerCommand type, const std::string& body) {
    switch (type) {
      case KVStoreServerProfilerCommand::kSetConfig:
        SetProfilerConfig(body.substr(0, body.size() - 1));
        break;
      case KVStoreServerProfilerCommand::kState:
        MXSetProfilerState(static_cast<int>(body.front() - '0'));
        break;
      case KVStoreServerProfilerCommand::kPause:
        MXProfilePause(static_cast<int>(body.front() - '0'));
        break;
      case KVStoreServerProfilerCommand::kDump:
        MXDumpProfile(static_cast<int>(body.front() - '0'));
        break;
    }
  }

  void SetProfilerConfig(std::string params_str) {
    std::vector<std::string> elems;
    mxnet::kvstore::split(params_str, ',', std::back_inserter(elems));
    std::vector<const char*> ckeys;
    std::vector<const char*> cvals;
    ckeys.reserve(elems.size());
    cvals.reserve(elems.size());

    for (size_t i=0; i < elems.size(); i++) {
      std::vector<std::string> parts;
      mxnet::kvstore::split(elems[i], ':', std::back_inserter(parts));
      CHECK_EQ(parts.size(), 2) << "Improper profiler config passed from worker";
      CHECK(!parts[0].empty()) << "ProfilerConfig parameter is empty";
      CHECK(!parts[1].empty()) << "ProfilerConfig value is empty for parameter "<< parts[0];
      if (parts[0] == "filename") {
        parts[1] = "rank" + std::to_string(ps::MyRank()) + "_" + parts[1];
      }
      char* ckey = new char[parts[0].length() + 1];
      std::snprintf(ckey, parts[0].length() + 1, "%s", parts[0].c_str());
      ckeys.push_back(ckey);

      char* cval = new char[parts[1].length() + 1];
      std::snprintf(cval, parts[1].length() + 1, "%s", parts[1].c_str());
      cvals.push_back(cval);
    }
    MXSetProfilerConfig(elems.size(), &ckeys[0], &cvals[0]);
    for (size_t i=0; i < ckeys.size(); i++) {
      delete[] ckeys[i];
      delete[] cvals[i];
    }
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<char>& req_data,
                    ps::KVServer<char>* server) {
    DataHandleType type = DepairDataHandleType(req_meta.cmd);
    switch (type.requestType) {
      case RequestType::kRowSparsePushPull:
        DataHandleRowSparse(type, req_meta, req_data, server);
        break;
      case RequestType::kCompressedPushPull:
        DataHandleCompressed(type, req_meta, req_data, server);
        break;
      case RequestType::kDefaultPushPull:
        DataHandleDefault(type, req_meta, req_data, server);
        break;
    }
  }

  inline bool has_multi_precision_copy(const DataHandleType type) {
    return multi_precision_ && type.dtype != mshadow::kFloat32;
  }

  inline void ApplyUpdates(const DataHandleType type, const uint64_t key,
                           UpdateBuf *update_buf, ps::KVServer<char>* server) {
    if (!sync_mode_ || update_buf->request.size() == (size_t) ps::NumWorkers()) {
      auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
      auto& update =  sync_mode_ ? update_buf->merged : update_buf->temp_array;
      // NOTE: not sure whether we need this WaitToRead, default is disabled
      if (update_buf_wait_) update_buf->merged.WaitToRead();

      // async mode does not need this Copy
      if (sync_mode_) CopyFromTo(update_buf->merged, &stored);

      if (has_multi_precision_copy(type)) CopyFromTo(stored, store_[key]);
      update_buf->request.clear();
    } 
  }

  void DecodeRowIds(const ps::SArray<ps::Key> &keys, int64_t *indices,
                    const int64_t master_key, const int64_t num_rows) {
    indices[0] = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
      uint64_t key = DecodeKey(keys[i]);
      auto row_id = key - master_key;
      indices[i - 1] = row_id;
    }
  }

  void AccumulateRowSparseGrads(const DataHandleType type,
                                const NDArray& recved,
                                UpdateBuf* updateBuf) {
    NDArray out(kRowSparseStorage, updateBuf->merged.shape(), Context(), true,
                has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
    if (has_multi_precision_copy(type)) CopyFromTo(recved, updateBuf->temp_array);
    const NDArray& to_merge = has_multi_precision_copy(type) ? updateBuf->temp_array : recved;
    // accumulate row_sparse gradients
    using namespace mshadow;
    Engine::Get()->PushAsync(
    [to_merge, updateBuf, out](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      op::ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(
      {}, {}, {to_merge, updateBuf->merged}, {kWriteTo}, {out});
      on_complete();
    }, to_merge.ctx(), {to_merge.var(), updateBuf->merged.var()}, {out.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    CopyFromTo(out, &(updateBuf->merged), 0);
    updateBuf->merged.WaitToRead();
  }

  void RowSparsePullResponse(const DataHandleType type,
                             const int master_key,
                             const size_t num_rows,
                             const ps::KVMeta& req_meta,
                             const ps::KVPairs<char>& req_data,
                             ps::KVServer<char>* server) {
    if (log_verbose_) LOG(INFO) << "pull: " << master_key;
    ps::KVPairs<char> response;
    if (num_rows == 0) {
      std::vector<int> lens(req_data.keys.size(), 0);
      response.keys = req_data.keys;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
      return;
    }
    const NDArray& stored = store_[master_key];
    if (has_multi_precision_copy(type)) stored.WaitToRead();
    CHECK(!stored.is_none()) << "init " << master_key << " first";
    auto shape = stored.shape();
    auto unit_len = shape.ProdShape(1, shape.ndim());
    const int num_bytes = mshadow::mshadow_sizeof(type.dtype);
    const int unit_size = unit_len * num_bytes;
    const char* data = static_cast<char *> (stored.data().dptr_);
    auto len = num_rows * unit_size;
    // concat values
    response.vals.resize(len);
    #pragma omp parallel for
    for (size_t i = 1; i <= num_rows; i++) {
      uint64_t key = DecodeKey(req_data.keys[i]);
      int64_t row_id = key - master_key;
      const auto src = data + row_id * unit_size;
      auto begin = (i - 1) * unit_size;
      auto end = i * unit_size;
      response.vals.segment(begin, end).CopyFrom(src, unit_size);
    }
    // setup response
    response.keys = req_data.keys;
    std::vector<int> lens(req_data.keys.size(), unit_len);
    lens[0] = 0;
    response.lens.CopyFrom(lens.begin(), lens.end());
    server->Response(req_meta, response);
  }

  void InitRowSparseStored(const DataHandleType type,
                           const int master_key,
                           const size_t num_rows,
                           const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    auto& stored = has_multi_precision_copy(type) ? store_realt_[master_key] : store_[master_key];
    int dtype = type.dtype;
    int num_bytes = mshadow::mshadow_sizeof(dtype);
    auto unit_len = req_data.lens[1] / num_bytes;
    CHECK_GT(unit_len, 0);
    size_t ds[] = {num_rows, (size_t) unit_len};
    TShape dshape(ds, ds + 2);
    CHECK_EQ(req_data.vals.size(), num_rows * unit_len * num_bytes);
    TBlob recv_blob;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()), dshape, cpu::kDevMask);
    })
    NDArray recved = NDArray(recv_blob, 0);
    stored = NDArray(kRowSparseStorage, dshape, Context(), true,
                     has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
    if (has_multi_precision_copy(type)) {
      store_[master_key] = NDArray(kRowSparseStorage, dshape, Context(), true, type.dtype);
    }
    Engine::Get()->PushAsync(
    [this, recved, stored, type](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      NDArray rsp = stored;
      stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      using namespace mxnet::op;
      nnvm::dim_t nnr = rsp.shape()[0];
      MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
        IType* idx = rsp.aux_data(rowsparse::kIdx).dptr<IType>();
        mxnet_op::Kernel<PopulateFullIdxRspKernel, cpu>::Launch(s, nnr, idx);
      });
      TBlob rsp_data = rsp.data();
      // copies or casts as appropriate
      ndarray::Copy<cpu, cpu>(recved.data(), &rsp_data, Context(), Context(), RunContext());
      on_complete();
    }, recved.ctx(), {recved.var()}, {stored.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    if (has_multi_precision_copy(type)) {
      CopyFromTo(stored, store_[master_key]);
      store_[master_key].WaitToRead();
    }
    stored.WaitToRead();
    server->Response(req_meta);
  }

  void DataHandleRowSparse(const DataHandleType type, const ps::KVMeta& req_meta,
                           const ps::KVPairs<char>& req_data,
                           ps::KVServer<char>* server) {
    uint64_t master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key];
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      if (stored.is_none()) {
        if (log_verbose_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        InitRowSparseStored(type, master_key, num_rows, req_meta, req_data, server);
        return;
      } else {
        if (log_verbose_) LOG(INFO) << "push: " << master_key << " " << req_data.keys;
        auto& updates = update_buf_[master_key];
        if (sync_mode_ && updates.merged.is_none()) {
          updates.merged = NDArray(kRowSparseStorage, stored.shape(), Context(), true,
                                   has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        }
        if (has_multi_precision_copy(type) && updates.temp_array.is_none()) {
          updates.temp_array = NDArray(kRowSparseStorage, stored.shape(), Context(), false,
                                       mshadow::kFloat32);
        }

        if (num_rows == 0) {
          if (sync_mode_) {
            if (updates.request.empty()) {
              // reset to zeros
              int merged_dtype = has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype;
              updates.merged = NDArray(kRowSparseStorage, stored.shape(), Context(),
                                       true, merged_dtype);
            }  // else nothing to aggregate
            updates.request.push_back(req_meta);
            ApplyUpdates(type, master_key, &updates, server);
          } else {
            server->Response(req_meta);
          }
        } else {
          auto unit_len = req_data.lens[1] / mshadow::mshadow_sizeof(type.dtype);
          CHECK_GT(unit_len, 0);
          // indices
          std::vector<int64_t> indices(num_rows);
          DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);

          // data
          TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
          size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
          TShape dshape(ds, ds + 2);
          TBlob recv_blob;
          MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
            recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()),
                              dshape, cpu::kDevMask);
          })
          // row_sparse NDArray
          NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

          if (updates.request.empty()) {
            if (sync_mode_) {
              CopyFromTo(recved, updates.merged);
            } else {
              if (has_multi_precision_copy(type)) {
                CopyFromTo(recved, updates.temp_array);
              } else {
                updates.temp_array = recved;
              }
            }
          } else {
            CHECK(sync_mode_);
            AccumulateRowSparseGrads(type, recved, &updates);
          }
          updates.request.push_back(req_meta);
          ApplyUpdates(type, master_key, &updates, server);
        }
      }
    } else {
      // pull
      RowSparsePullResponse(type, master_key, num_rows, req_meta, req_data, server);
    }
  }

  std::unordered_map<int, ps::KVPairs<char> > server_response_map;

  void DefaultStorageResponse(const DataHandleType type,
                              const int key,
                              const ps::KVMeta& req_meta,
                              const ps::KVPairs<char> &req_data,
                              ps::KVServer<char>* server) {
    const NDArray& stored = store_[key];
    CHECK(!stored.is_none()) << "init " << key << " first";
    // as server returns when store_realt is ready in this case
    if (has_multi_precision_copy(type)) stored.WaitToRead();
    auto len = stored.shape().Size() * mshadow::mshadow_sizeof(stored.dtype());

    // send pull response
    auto iterator = server_response_map.find(key);
    if (iterator==server_response_map.end()) { // new key
      ps::KVPairs<char> response;
      response.keys = req_data.keys;
      response.lens = {len};

      stored.WaitToRead();
      if(enable_pull_zero_copy_) {
        response.vals = ps::SArray<char>(static_cast<char*>(stored.data().dptr_), len, false); // enable zero copy
      }
      else {
        response.vals.CopyFrom(static_cast<const char*>(stored.data().dptr_), len);
      }
      server_response_map[key] = response; // add to the map
      server->Response(req_meta, response);
    }
    else { // not new key, then reuse the memory address to avoid ibv_reg_mr on RDMA data path
      ps::KVPairs<char> *response = &iterator->second;
      // keys and lens remain unchanged, just update vals
      if(enable_pull_zero_copy_) {
        response->vals = ps::SArray<char>(static_cast<char*>(stored.data().dptr_), len, false); // enable zero copy
      }
      else {
        response->vals.CopyFrom(static_cast<const char*>(stored.data().dptr_), len);
      }
      Engine::Get()->PushAsync(
            [this, server, &req_meta, response](RunContext ctx, Engine::CallbackOnComplete on_complete) {
              server->Response(req_meta, *response);
              on_complete();
            }, stored.ctx(), {stored.var()}, {},
            FnProperty::kCPUPrioritized, 0, "BYTEPS_SEND_PULL_RESPONSE");
    }
  }

  void DataHandleCompressed(const DataHandleType type,
                            const ps::KVMeta& req_meta,
                            const ps::KVPairs<char> &req_data,
                            ps::KVServer<char>* server) {
    CHECK_EQ(type.dtype, mshadow::kFloat32)
      << "Gradient compression is currently supported for fp32 only";
    if (req_meta.push) {
      // there used several WaitToRead, this is because \a recved's memory
      // could be deallocated when this function returns. so we need to make sure
      // the operators with \a NDArray are actually finished

      // first for dummy key which represents original size of array, whose len is 0
      CHECK_EQ(req_data.keys.size(), (size_t)2);
      CHECK_EQ(req_data.lens.size(), (size_t)2);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[1]);

      uint64_t original_size = DecodeKey(req_data.keys[0]);
      uint64_t key = DecodeKey(req_data.keys[1]);
      auto& stored = store_[key];

      size_t ds[] = {(size_t)req_data.lens[1] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob(reinterpret_cast<real_t*>(req_data.vals.data()), dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

      NDArray decomp_buf = decomp_buf_[key];
      dshape = TShape{(int64_t) original_size};

      if (decomp_buf.is_none()) {
        decomp_buf = NDArray(dshape, Context());
      }

      if (stored.is_none()) {
        stored = NDArray(dshape, Context());
        gradient_compression_->Dequantize(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = update_buf_[key];
        if (merged.merged.is_none()) {
          merged.merged = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          gradient_compression_->Dequantize(recved, &merged.merged, 0);
        } else {
          gradient_compression_->Dequantize(recved, &decomp_buf, 0);
          merged.merged += decomp_buf;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(type, key, &merged, server);
      } else {
        // async push
        gradient_compression_->Dequantize(recved, &decomp_buf, 0);
        exec_.Exec([this, key, &decomp_buf, &stored]() {
          CHECK(updater_);
          updater_(key, decomp_buf, &stored);
        });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {       // pull
      CHECK_EQ(req_data.keys.size(), (size_t)1);
      CHECK_EQ(req_data.lens.size(), (size_t)0);
      // temporarily comment this two lines, should revisit here if we use compression
      // uint64_t key = DecodeKey(req_data.keys[0]);
      // DefaultStorageResponse(type, key, req_meta, req_data, server);
    }
  }

  void SendPushResponse(uint64_t key, const ps::KVMeta& req, ps::KVServer<char>* server){
    auto iterator = push_response_map.find(key);
    if (iterator==push_response_map.end()){ // new key
      ps::KVPairs<char> response;
      response.keys.push_back(key);
      push_response_map[key] = response; // add to the map
      server->Response(req, response);
    }
    else{ // not new key, then reuse the memory address to avoid ibv_reg_mr on RDMA data path
      ps::KVPairs<char> *response = &iterator->second;
      response->keys[0]=key;
      server->Response(req, *response);
    }
  }

  void DataHandleDefault(const DataHandleType type, const ps::KVMeta& req_meta,
                         const ps::KVPairs<char> &req_data,
                         ps::KVServer<char>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
      if (log_key_info_) LOG(INFO) << "push key=" << DecodeKey(req_data.keys[0])
                                   << "\t sender=" << req_meta.sender
                                   << "\t size=" << (size_t) req_data.lens[0];
    } else {
      if (log_key_info_) LOG(INFO) << "pull key=" << (uint64_t) DecodeKey(req_data.keys[0])
                                   << "\t sender=" << req_meta.sender;
    }
    uint64_t key = DecodeKey(req_data.keys[0]);
    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
      size_t ds[] = {(size_t) req_data.lens[0] / mshadow::mshadow_sizeof(type.dtype)};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob;
      MSHADOW_TYPE_SWITCH(type.dtype, DType, {
        recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()), dshape, cpu::kDevMask);
      })
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // buffer the request meta
        auto &updates = update_buf_[key];
        updates.request.push_back(req_meta);

        if (updates.request.size() < (size_t) ps::NumWorkers()) return;

        if (log_key_info_) {
          LOG(INFO) << "Collected all " << updates.request.size()
                    << " requests for key=" << key
                    << ", init the store buffer size=" << (size_t) req_data.lens[0];
        }

        // initialization
        stored = NDArray(dshape, Context(), false,
                         has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        CopyFromTo(recved, &stored, 0);
        if (has_multi_precision_copy(type)) {
          auto& stored_dtype = store_[key];
          stored_dtype = NDArray(dshape, Context(), false, type.dtype);
          CopyFromTo(stored, stored_dtype);
          stored_dtype.WaitToRead();
        }
        // delay sending response until stored is ready
        stored.WaitToRead();

        for (const auto& req : updates.request) {
          SendPushResponse(key, req, server);
        }

        updates.request.clear();
      } else {
        auto &updates = update_buf_[key];
        if (sync_mode_ && updates.merged.is_none()) {
          updates.merged = NDArray(dshape, Context(), false,
                                   has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        }
        if (has_multi_precision_copy(type) && updates.temp_array.is_none()) {
          updates.temp_array = NDArray(dshape, Context(), false, mshadow::kFloat32);
        }
        if (updates.request.empty()) { // from the first incoming worker
          if (sync_mode_) {
            CopyFromTo(recved, updates.merged);
          } else {
            if (has_multi_precision_copy(type)) {
              CopyFromTo(recved, updates.temp_array);
            } else {
              stored += recved;
            }
          }
        } else { // from other workers
          CHECK(sync_mode_);
          if (has_multi_precision_copy(type)) {
            CopyFromTo(recved, updates.temp_array);
            updates.merged += updates.temp_array;
          } else {
            Engine::Get()->PushAsync(
            [this, updates, recved](RunContext ctx, Engine::CallbackOnComplete on_complete) {
              CHECK_GE(bps_reducer_.sum(bps_reducer_.GetData(&updates.merged), bps_reducer_.GetData(&recved),
                                        bps_reducer_.GetSize(&recved), bps_reducer_.GetDType(&recved)), 0);
              on_complete();
            }, updates.merged.ctx(), {recved.var()}, {updates.merged.var()},
            FnProperty::kCPUPrioritized, 0, "BYTEPS_SUMMATION");
          }
        }
        // add a worker information (request.size() is the # workers received)
        updates.request.push_back(req_meta);
        SendPushResponse(key, req_meta, server);
        ApplyUpdates(type, key, &updates, server);
      }
    } else {
      auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
      if (stored.is_none()) {
        CHECK(0) << "Processing pull request when the NDArray of key " << key << " has not been inited yet, which is not expected.";
      }

      // process pull request
      auto &updates = update_buf_[key];
      DefaultStorageResponse(type, key, req_meta, req_data, server);
    }
  }

  uint64_t DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined mode for push
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  /**
   * \brief store_ contains the value at kvstore for each key
   */
  std::unordered_map<uint64_t, NDArray> store_;
  std::unordered_map<uint64_t, NDArray> store_realt_;

  /**
   * \brief merge_buf_ is a buffer used if sync_mode is true. It represents
   * values from different workers being merged. The store will be updated
   * to this value when values from all workers are pushed into this buffer.
   */
  std::unordered_map<uint64_t, UpdateBuf> update_buf_;

  /**
   * \brief decomp_buf_ is a buffer into which compressed values are
   * decompressed before merging to the store. used when compress_!='none'
   */
  std::unordered_map<uint64_t, NDArray> decomp_buf_;

  Executor exec_;
  ps::KVServer<char>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_;

  // whether to LOG key trace
  bool log_key_info_;

  // whether to enable zero copy for pull request
  bool enable_pull_zero_copy_;
  /*
   * \brief whether to use multi precision mode.
   * in multi precision mode, all weights are stored as float32.
   * any gradient received will be cast to float32 before accumulation and updating of weights.
   */
  bool multi_precision_;

  bool update_buf_wait_;

  CpuReducer bps_reducer_;

  /*
   * send push response with the key as value
   */
  std::unordered_map<uint64_t, ps::KVPairs<char> > push_response_map;

  /**
   * \brief gradient compression object.
   * starts with none, used after SetGradientCompression sets the type
   * currently there is no support for unsetting gradient compression
   */
  std::shared_ptr<kvstore::GradientCompression> gradient_compression_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
