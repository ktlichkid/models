#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from paddle.fluid.op import Operator, DynamicRecurrentOp
import paddle.fluid.core as core
import unittest
import numpy as np


def create_tensor(scope, name, np_data):
    tensor = scope.var(name).get_tensor()
    #tensor.set(np_data, core.CPUPlace())
    tensor.set(np_data, core.CUDAPlace(0))
    return tensor


class BeamSearchOpTester(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        #self.tensor = None
        #self._create_ids()
        #self._create_scores()
        #self._create_pre_ids()
        #self.scope.var('selected_ids')
        #self.scope.var('selected_scores')

    def test_set(self):
        tensor_pre = self.scope.var("pre_ids").get_tensor()
        np_data_pre = np.array([[1, 2, 3, 4]], dtype='int64')
        tensor_pre.set(np_data_pre, core.CUDAPlace(0))
        print "Check point 1"    

        tensor_ids = self.scope.var("ids").get_tensor()
        lod_ids = [[0, 1, 4], [0, 1, 2, 3, 4]]
        np_data_ids = np.array(
            [[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]], dtype='int64')
        tensor_ids.set(np_data_ids, core.CUDAPlace(0))
        tensor_ids.set_lod(lod_ids)
        print "Check point 2"

        tensor_score = self.scope.var("scores").get_tensor()
        np_data_score = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.9, 0.5, 0.1],
                [0.7, 0.5, 0.1],
            ],
            dtype='float32')
        tensor_score.set(np_data_score, core.CUDAPlace(0))
        tensor_score.set_lod(lod_ids)
        print "Check point 3"
        


    #def test_run(self):
    #    op = Operator(
    #        'beam_search',
    #        pre_ids="pre_ids",
    #        ids='ids',
   #         scores='scores',
    #        selected_ids='selected_ids',
    #        selected_scores='selected_scores',
    #        level=0,
    #        beam_size=2,
    #        end_id=0, )
        #op.run(self.scope, core.CPUPlace())
    #    op.run(self.scope, core.CUDAPlace(0))
      #  selected_ids = self.scope.find_var("selected_ids").get_tensor()
      #  print 'selected_ids', np.array(selected_ids)
      #  print 'lod', selected_ids.lod()

    def _create_pre_ids(self):
        np_data = np.array([[1, 2, 3, 4]], dtype='int64')
        tensor = create_tensor(self.scope, "pre_ids", np_data)

    def _create_ids(self):
        self.lod = [[0, 1, 4], [0, 1, 2, 3, 4]]
        np_data = np.array(
            [[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]], dtype='int64')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.9, 0.5, 0.1],
                [0.7, 0.5, 0.1],
            ],
            dtype='float32')
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)


if __name__ == '__main__':
    unittest.main()
