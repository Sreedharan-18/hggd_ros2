from setuptools import setup
import os
from glob import glob

package_name = "grasp_pose_detection"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "srv"), glob("srv/*.srv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@todo.todo",
    description="HGGD grasp service wrapper",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hggd_grasp_service_node = grasp_pose_detection.grasp_pose_service:main",
        ],
    },
)
