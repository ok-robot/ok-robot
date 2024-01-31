from typing import Optional, List, Union

import zmq
import numpy as np
from utils.types import Number

class ZmqSocket:
    def __init__(self, cfgs):
        # init socket with port number
        zmq_context = zmq.Context()
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:" + str(cfgs.port))

    def send_array(
        self, 
        data: np.ndarray, 
        flags: int = 0, 
        copy: bool = True, 
        track: bool = False
    ) ->  Optional[int]:
        """send a numpy array with metadata"""
        md = dict(
            dtype = str(data.dtype),
            shape = data.shape,
        )
        self.socket.send_json(md, flags|zmq.SNDMORE)
        
        return self.socket.send(np.ascontiguousarray(data), flags, copy=copy, track=track)

    def recv_array(
        self,
        flags: int = 0,
        copy: bool = True,
        track: bool = False
    ) -> np.ndarray:
        """Receive a NumPy array."""
        md = self.socket.recv_json(flags=flags)
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        data = np.frombuffer(msg, dtype=md['dtype'])

        return data.reshape(md['shape'])
    
    def send_data(
        self,
        data: Union[str, Union[List[Number], List[Union[List[Number], str]]]]
    ) -> Optional[bool]:
        """Send msg - string or list of Numbers or list of list Numbers or strings """

        # After sending anytype of data other than str it waits for the string confirmation from robot
        if isinstance(data, str):
            self.socket.send_string(data)
        elif isinstance(data, list) and all((not isinstance(num, list)) for num in data):
            data = np.array(data)
            self.send_array(data)
            print(self.recv_string())
        else:
            for d in data:
                if isinstance(d, str):
                    self.socket.send_string(d)
                else:
                    print(d)
                    data = np.array(d)
                    self.send_array(data)
                    print(self.recv_string())                    
    
    def recv_string(self) -> str:
        return self.socket.recv_string()
        