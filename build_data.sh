#!bin/bash

set -m
module load conda

mkcd1() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data1
}

mkcd2() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data2
}

mkcd3() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data3
}

mkcd4() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data4
}

mkcd5() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data5
}

mkcd6() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data6
}
mkcd7() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data7
}
mkcd8() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data8
}
mkcd9() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data9
}
mkcd10() {
    conda activate /blue/rcstudents/jrockholt/test_env
    python build_data.py rrt_data10
}

mkcd1 &
mkcd2 &
mkcd3 &
mkcd4 &
mkcd5 &
mkcd6 &
mkcd7 &
mkcd8 &
mkcd9 &
mkcd10 &
wait
