#!/bin/bash
LAMBDA=0.00000001

./bench_OptmSync --population 8388608 --nodes-per-lp-log 7 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 20 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 8388608 --nodes-per-lp-log 6 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 20 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 8388608 --nodes-per-lp-log 5 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 20 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 8388608 --nodes-per-lp-log 4 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 20 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 8388608 --nodes-per-lp-log 3 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 20 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 4194304 --nodes-per-lp-log 7 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 10 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 4194304 --nodes-per-lp-log 6 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 10 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 4194304 --nodes-per-lp-log 5 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 10 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 4194304 --nodes-per-lp-log 4 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 10 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 4194304 --nodes-per-lp-log 3 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 10 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 2097152 --nodes-per-lp-log 7 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 5 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 2097152 --nodes-per-lp-log 6 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 5 --runs 1 --warmup 100 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 2097152 --nodes-per-lp-log 5 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 5 --runs 1 --warmup 100 --committed 200 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 2097152 --nodes-per-lp-log 4 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 5 --runs 1 --warmup 100 --committed 200 --lookahead 1.0 --lambda ${LAMBDA}
./bench_OptmSync --population 2097152 --nodes-per-lp-log 3 --events-per-node 8 --states-per-node 10 --antimsgs-per-node 10 --committed 5 --runs 1 --warmup 100 --committed 200 --lookahead 1.0 --lambda ${LAMBDA}
