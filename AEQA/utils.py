# utils.py

import os
import json
from typing import Iterator, Tuple, List
import quaternion  # 确保已安装 numpy-quaternion 包
# from your_module import AgentState, get_shortest_path  # 根据实际情况导入
from habitat_sim.agent.agent import AgentState, SixDOFPose
import os
from typing import Iterator, List

class DestinationDataLoader:
    def __init__(self, destinations_dir: str):
        """
        初始化数据加载器。

        :param destinations_dir: 'destinations' 目录的路径。
        """
        self.destinations_dir = destinations_dir
        self.file_paths = self._get_json_file_paths()

    def _get_json_file_paths(self) -> List[str]:
        """
        获取 destinations 目录下所有 JSON 文件的路径。

        :return: JSON 文件路径的列表。
        """
        json_files = [
            os.path.join(self.destinations_dir, file)
            for file in os.listdir(self.destinations_dir)
            if file.endswith('.json')
        ]
        if not json_files:
            print(f"未在目录中找到 JSON 文件: {self.destinations_dir}")
        return json_files

    def __iter__(self) -> Iterator[str]:
        """
        使数据加载器可迭代，逐个返回 JSON 文件路径。

        :return: JSON 文件路径的迭代器。
        """
        for file_path in self.file_paths:
            yield file_path