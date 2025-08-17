# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import struct


# ... imports unchanged ...

def parse_raw_adc(source_fp, dest_fp):
    buff = np.fromfile(source_fp, dtype=np.uint8)
    buff_pos = 0
    adc_data = []
    first_seq = None
    HEADER_SIZE = 16   # not 14

    while buff_pos < len(buff):
        # header
        seq_bytes = buff[buff_pos:buff_pos+4].tobytes()
        len_bytes = buff[buff_pos+4:buff_pos+8].tobytes()
        packet_num    = struct.unpack('<I', seq_bytes)[0]      # unsigned
        packet_length = struct.unpack('<I', len_bytes)[0]      # unsigned
        buff_pos += HEADER_SIZE

        # normalize to zero-based relative index
        if first_seq is None:
            first_seq = packet_num
        idx = packet_num - first_seq
        if idx < 0:
            # handle starting mid-stream / wrap
            first_seq = packet_num
            idx = 0

        pkt = buff[buff_pos:buff_pos + packet_length]
        buff_pos += packet_length

        # ensure capacity
        while len(adc_data) < idx:
            adc_data.append(np.zeros(packet_length, dtype=np.uint8))

        if len(adc_data) == idx:
            adc_data.append(pkt)
        else:
            adc_data[idx] = pkt

    adc_data = np.concatenate(adc_data)
    with open(dest_fp, 'wb') as fp:
        fp.write(adc_data.tobytes())
