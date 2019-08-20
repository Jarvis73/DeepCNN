# Copyright 2019 Jianwei Zhang All Right Reserved.
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
#
# =================================================================================


class Accumulator(object):
    def __init__(self, with_value=True):
        self._count = 0
        self._value = 0.
        self.with_value = with_value

    def reset(self):
        self._count = 0
        self._value = 0.

    @property
    def avg(self):
        if self._count > 0:
            return self._value / self._count
        return 0.

    def pop(self):
        avg = self.avg
        self.reset()
        return avg

    def update(self, new_value=None):
        if self.with_value:
            self._value += new_value
        self._count += 1

    def updates(self, values):
        self._value += sum(values)
        self._count += len(values)


class MovingAverage(object):
    """
    Perform moving average accumulation

    v_t+1 = alpha * vt + (1 - alpha) * new_val
    """
    def __init__(self, alpha=0.9, skip_first=True):
        self._count = 0
        self._value = 0.
        self.alpha = alpha
        self.skip_first = skip_first

    def reset(self):
        self._count = 0
        self._value = 0.

    @property
    def value(self):
        return self._value

    @property
    def count(self):
        return self._count

    def update(self, new_value=None):
        if self._count == 0 and self.skip_first:
            self._value = new_value
        else:
            self._value = self.alpha * self._value + (1 - self.alpha) * new_value
        self._count += 1
