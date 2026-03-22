/*  Benchmark wrapper for GPUTW PHOLD simulation.
    No Nelder-Mead tuning — all parameters fixed via command-line flags.
    Designed for fair comparison with WindowRacer. */

#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "statistics.cu"

/* Global variables */
__device__ uint		g_n_nodes;
__device__ uint		g_n_lps;
__device__ uint		g_nodes_per_lp;

/* Stats counters (defined in kernels.cu) */
extern __device__ uint  g_n_total_processed;
extern __device__ uint  g_n_total_rolledback;
extern __device__ char  g_track_stats;

static	uint	nodes_per_lp;
static	uint	n_nodes;
static	uint	n_lps;

static	uint	events_per_node;
static	uint	states_per_node;
static	uint	antimsgs_per_node;

static	float	committed_events_threshold;
static	float	inactive_lps_percent;
static	double	window_size;

static	uint	threads_per_block;
static	uint	n_blocks;

/* Private functions */
size_t	get_free_memory();
uint	get_number_blocks(uint n_threads);
double	get_gvt(double *d_ts_temp);

/* --- Argument parsing --- */

struct BenchParams {
	/* PHOLD model */
	int    population;         // number of nodes
	double lookahead;          // lookahead added to timestamps
	double mean;               // mean of exponential dist (= 1/lambda)

	/* Simulation control */
	double window;             // window_size for optimistic sync (negative = unlimited)
	float  inactive_percent;   // fraction of LPs that must be inactive before GVT advance
	int    nodes_per_lp_log;   // log2(nodes_per_lp)
	int    events_per_node;    // event queue capacity per node
	int    states_per_node;    // state queue capacity per node
	int    antimsgs_per_node;  // antimsg queue capacity per node
	int    zero_delay_pct;     // zero-delay ratio as 0-100 percent

	/* Run control */
	float  committed_mil;      // millions of committed events before stop
	int    n_runs;             // number of test runs
	int    warmup_ms;          // warmup time in ms
	int    measure_ms;         // measurement window in ms
	int    count_rollbacks;    // enable rollback/processed counting
	int    minimal;            // minimal mode: no sampling, no stats, just final MEv/s
};

static void print_usage(const char *prog) {
	printf("Usage: %s [options]\n", prog);
	printf("Options:\n");
	printf("  --population N      Number of nodes (default: 65536)\n");
	printf("  --lambda F          Exponential rate param; mean = 1/lambda (default: 0.0001 => mean=10000)\n");
	printf("  --mean F            Directly set mean (overrides --lambda)\n");
	printf("  --lookahead F       Lookahead value (default: 1.0)\n");
	printf("  --window F          Window size, negative for unlimited (default: 1000)\n");
	printf("  --inactive F        Inactive LP fraction 0.0-1.0 (default: 0.6)\n");
	printf("  --nodes-per-lp-log N  log2(nodes_per_lp) (default: 6 => 64)\n");
	printf("  --events-per-node N Event queue capacity per node (default: 64)\n");
	printf("  --states-per-node N State queue capacity per node (default: 30)\n");
	printf("  --antimsgs-per-node N Antimsg queue capacity per node (default: 30)\n");
	printf("  --zero-delay-pct N  Zero-delay event percentage 0-100 (default: 0)\n");
	printf("  --committed F       Millions of committed events (default: 1000)\n");
	printf("  --runs N            Number of test runs (default: 3)\n");
	printf("  --warmup N          Warmup time in ms (default: 200)\n");
	printf("  --measure N         Measurement window in ms (default: 300)\n");
	printf("  --count-rollbacks   Enable processed/rollback counters (has overhead)\n");
	printf("  --minimal           Minimal mode: no sampling, no stats, just final MEv/s\n");
	printf("  --help              Show this help\n");
}

static BenchParams parse_args(int argc, char *argv[]) {
	BenchParams p;
	p.population        = 65536;
	p.lookahead          = 1.0;
	p.mean               = 10000.0;
	p.window             = 1000.0;
	p.inactive_percent   = 0.6;
	p.nodes_per_lp_log   = 6;
	p.events_per_node    = 64;
	p.states_per_node    = 30;
	p.antimsgs_per_node  = 30;
	p.zero_delay_pct     = 0;
	p.committed_mil      = 1000;
	p.n_runs             = 3;
	p.warmup_ms          = 200;
	p.measure_ms         = 300;
	p.count_rollbacks    = 0;
	p.minimal            = 0;

	int mean_set = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0) {
			print_usage(argv[0]);
			exit(0);
		} else if (strcmp(argv[i], "--population") == 0 && i+1 < argc) {
			p.population = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--lambda") == 0 && i+1 < argc) {
			double lambda = atof(argv[++i]);
			if (!mean_set && lambda > 0) {
				p.mean = 1.0 / lambda;
			}
		} else if (strcmp(argv[i], "--mean") == 0 && i+1 < argc) {
			p.mean = atof(argv[++i]);
			mean_set = 1;
		} else if (strcmp(argv[i], "--lookahead") == 0 && i+1 < argc) {
			p.lookahead = atof(argv[++i]);
		} else if (strcmp(argv[i], "--window") == 0 && i+1 < argc) {
			p.window = atof(argv[++i]);
		} else if (strcmp(argv[i], "--inactive") == 0 && i+1 < argc) {
			p.inactive_percent = atof(argv[++i]);
		} else if (strcmp(argv[i], "--nodes-per-lp-log") == 0 && i+1 < argc) {
			p.nodes_per_lp_log = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--events-per-node") == 0 && i+1 < argc) {
			p.events_per_node = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--states-per-node") == 0 && i+1 < argc) {
			p.states_per_node = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--antimsgs-per-node") == 0 && i+1 < argc) {
			p.antimsgs_per_node = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--zero-delay-pct") == 0 && i+1 < argc) {
			p.zero_delay_pct = atoi(argv[++i]);
			if (p.zero_delay_pct < 0) p.zero_delay_pct = 0;
			if (p.zero_delay_pct > 100) p.zero_delay_pct = 100;
		} else if (strcmp(argv[i], "--committed") == 0 && i+1 < argc) {
			p.committed_mil = atof(argv[++i]);
		} else if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) {
			p.n_runs = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) {
			p.warmup_ms = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--measure") == 0 && i+1 < argc) {
			p.measure_ms = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--count-rollbacks") == 0) {
			p.count_rollbacks = 1;
		} else if (strcmp(argv[i], "--minimal") == 0) {
			p.minimal = 1;
		} else {
			printf("Unknown option: %s\n", argv[i]);
			print_usage(argv[0]);
			exit(1);
		}
	}

	return p;
}

/* --- Main --- */

int main(int argc, char *argv[]) {

	BenchParams bp = parse_args(argc, argv);

	/* Print config */
	printf("=== GPUTW PHOLD Benchmark ===\n");
	printf("Sync mode:       %s\n",
#if (OPTM_SYNC == 1)
		"Optimistic"
#else
		"Conservative"
#endif
	);
	printf("Allow ME:        %s\n",
#if (ALLOW_ME == 1)
		"yes"
#else
		"no"
#endif
	);
	printf("population:      %d\n", bp.population);
	printf("mean (1/lambda): %.6f\n", bp.mean);
	printf("lambda:          %.6f\n", 1.0 / bp.mean);
	printf("lookahead:       %.6f\n", bp.lookahead);
	printf("window_size:     %.6f\n", bp.window);
	printf("inactive_pct:    %.2f\n", bp.inactive_percent);
	printf("nodes_per_lp:    %d (2^%d)\n", 1 << bp.nodes_per_lp_log, bp.nodes_per_lp_log);
	printf("events_per_node: %d\n", bp.events_per_node);
	printf("states_per_node: %d\n", bp.states_per_node);
	printf("antimsgs_per_node: %d\n", bp.antimsgs_per_node);
	printf("zero_delay_pct:  %d%%\n", bp.zero_delay_pct);
	printf("committed target: %.0f M events\n", bp.committed_mil);
	printf("runs:            %d\n", bp.n_runs);
	printf("warmup:          %d ms\n", bp.warmup_ms);
	printf("measure:         %d ms\n", bp.measure_ms);
	printf("count_rollbacks: %s\n", bp.count_rollbacks ? "yes" : "no");
	printf("=============================\n\n");

	/* Setup fixed parameters */
	nodes_per_lp = 1 << bp.nodes_per_lp_log;
	n_nodes = bp.population;
	n_lps = n_nodes / nodes_per_lp;

	events_per_node   = bp.events_per_node;
	states_per_node   = bp.states_per_node;
	antimsgs_per_node = bp.antimsgs_per_node;

	committed_events_threshold = bp.committed_mil * 1000000.0f;
	inactive_lps_percent = bp.inactive_percent;
	window_size = bp.window;

#if (OPTM_SYNC == 0)
	states_per_node = 0;
	antimsgs_per_node = 0;
#endif

#if (ALLOW_ME == 0)
	inactive_lps_percent = 0;
#endif

	threads_per_block = 256;
	n_blocks = get_number_blocks(n_lps);

	double model_params[] = {(double)n_nodes, bp.lookahead, bp.mean, (double)bp.zero_delay_pct};
	uint n_params = 4;

	float results_rate[50];
	int n_runs = bp.n_runs;
	if (n_runs > 50) n_runs = 50;

	for (int r = 0; r < n_runs; r++) {

		double	*d_model_params;
		double	*d_lookahead;
		double	*d_ts_temp;
		uint	*d_n_events_cmt;
		uint	*d_inac_1, *d_inac_2, *d_inac_3;
		uint	*d_inac_4, *d_inac_5, *d_inac_6;
		char	*d_rollback_performed;

		double	h_lookahead;
		uint	h_n_events_cmt;
		uint	h_inac_1, h_inac_2, h_inac_3;
		uint	h_inac_4, h_inac_5, h_inac_6;
		char	h_rollback_performed;

		cudaDeviceReset();

		/* Enable/disable stats counters */
		{
			char flag = bp.count_rollbacks ? 1 : 0;
			uint zero = 0;
			cudaMemcpyToSymbol(g_track_stats, &flag, sizeof(char));
			cudaMemcpyToSymbol(g_n_total_processed, &zero, sizeof(uint));
			cudaMemcpyToSymbol(g_n_total_rolledback, &zero, sizeof(uint));
		}

		if (
		cudaMalloc(&d_model_params, sizeof(double) * n_params) != cudaSuccess ||
		cudaMalloc(&d_lookahead,	sizeof(double))	!= cudaSuccess ||
		cudaMalloc(&d_ts_temp,	sizeof(double) * n_blocks) != cudaSuccess ||
		cudaMalloc(&d_n_events_cmt,	sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_1,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_2,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_3,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_4,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_5,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_inac_6,		sizeof(uint))	!= cudaSuccess ||
		cudaMalloc(&d_rollback_performed, sizeof(char))	!= cudaSuccess) {
			printf("ERROR: cudaMalloc failed.\n");
			return 1;
		}

		cudaMemcpy(d_model_params, model_params, sizeof(double) * n_params,
			cudaMemcpyHostToDevice);

		h_n_events_cmt = 0;
		cudaMemcpy(d_n_events_cmt, &h_n_events_cmt, sizeof(uint),
			cudaMemcpyHostToDevice);

		h_rollback_performed = 0;
		cudaMemcpy(d_rollback_performed, &h_rollback_performed, sizeof(char),
			cudaMemcpyHostToDevice);

		char res = 0;
		res += malloc_nodes(n_nodes);
		res += malloc_queues(n_nodes,
			events_per_node, states_per_node, antimsgs_per_node);

		if (res != 2) {
			free_nodes();
			free_queues();
			printf("ERROR: Memory not enough.\n");
			return 1;
		}

		/* Initialization */
		kernel_set_params<<<1,1>>>(
			n_nodes, n_lps, nodes_per_lp,
			events_per_node, states_per_node, antimsgs_per_node,
			d_model_params, n_params);

		kernel_get_lookahead<<<1,1>>>(d_lookahead);
		cudaMemcpy(&h_lookahead, d_lookahead, sizeof(double),
			cudaMemcpyDeviceToHost);

		kernel_init_queues<<<n_blocks, threads_per_block>>>();
		kernel_init_nodes<<<n_blocks, threads_per_block>>>();
		kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();

		/* Timing */
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float time_ms;

		float total_events = 0;
		float final_rate;

		cudaDeviceSynchronize();
		cudaEventRecord(start);

	if (bp.minimal) {
		/* ---- MINIMAL MODE: no sampling, no sync, just run ---- */
		while (1) {
			double gvt = get_gvt(d_ts_temp);

			h_n_events_cmt = 0;
			cudaMemcpy(d_n_events_cmt, &h_n_events_cmt, sizeof(uint),
				cudaMemcpyHostToDevice);

			kernel_clean_queues<<<n_blocks, threads_per_block>>>(
				gvt, d_n_events_cmt);

			cudaMemcpy(&h_n_events_cmt, d_n_events_cmt, sizeof(uint),
				cudaMemcpyDeviceToHost);
			total_events += h_n_events_cmt;

			if (total_events > committed_events_threshold) {
				break;
			}

			while (1) {
				h_inac_1 = h_inac_2 = h_inac_3 = 0;
				h_inac_4 = h_inac_5 = h_inac_6 = 0;
				cudaMemcpy(d_inac_1, &h_inac_1, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_2, &h_inac_2, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_3, &h_inac_3, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_4, &h_inac_4, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_5, &h_inac_5, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_6, &h_inac_6, sizeof(uint),
					cudaMemcpyHostToDevice);

#if (OPTM_SYNC == 1)
				kernel_handle_next_event<<<
					n_blocks, threads_per_block>>>(
						gvt, window_size,
						d_inac_1, d_inac_2, d_inac_3,
						d_inac_4, d_inac_5, d_inac_6);
#else
				kernel_handle_next_event<<<
					n_blocks, threads_per_block>>>(
						gvt, h_lookahead,
						d_inac_1, d_inac_2, d_inac_3,
						d_inac_4, d_inac_5, d_inac_6);
#endif

				cudaMemcpy(&h_inac_1, d_inac_1, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_2, d_inac_2, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_3, d_inac_3, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_4, d_inac_4, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_5, d_inac_5, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_6, d_inac_6, sizeof(uint),
					cudaMemcpyDeviceToHost);

#if (ALLOW_ME == 1)
				uint inac =
					h_inac_1 + h_inac_2 + h_inac_3 +
					h_inac_4 + h_inac_5 + h_inac_6;
				if (inac >= n_lps * inactive_lps_percent) { break; }
#else
				break;
#endif
			}

#if (OPTM_SYNC == 1)
			while (1) {
				h_rollback_performed = 0;
				cudaMemcpy(d_rollback_performed, &h_rollback_performed,
					sizeof(char), cudaMemcpyHostToDevice);

				kernel_roll_back<<<n_blocks, threads_per_block>>>(
					d_rollback_performed);

				cudaMemcpy(&h_rollback_performed, d_rollback_performed,
					sizeof(char), cudaMemcpyDeviceToHost);
				if (h_rollback_performed == 0) { break; }
			}
#endif

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("FATAL ERROR: %s\n", cudaGetErrorString(err));
				return 1;
			}

			kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_ms, start, stop);

		final_rate = total_events / time_ms / 1000.0f;
		printf("  Run %d: %.3f MEv/s (%.0f events, %.0f ms)\n",
			r, final_rate, total_events, time_ms);

	} else {
		/* ---- FULL MODE: sampling, %activity, optional rollback stats ---- */
		cudaEvent_t mstart, mstop;
		cudaEventCreate(&mstart);
		cudaEventCreate(&mstop);

		float measure_events = 0;
		char  measuring = 0;
		int   n_measurements = 0;
		float sum_rates = 0;
		uint64_t total_execute_cycles = 0;

		cudaEventRecord(mstart);

		while (1) {
			double gvt = get_gvt(d_ts_temp);

			h_n_events_cmt = 0;
			cudaMemcpy(d_n_events_cmt, &h_n_events_cmt, sizeof(uint),
				cudaMemcpyHostToDevice);

			kernel_clean_queues<<<n_blocks, threads_per_block>>>(
				gvt, d_n_events_cmt);

			cudaMemcpy(&h_n_events_cmt, d_n_events_cmt, sizeof(uint),
				cudaMemcpyDeviceToHost);
			total_events += h_n_events_cmt;

			if (total_events > committed_events_threshold) {
				break;
			}

			/* Measurement windows */
			cudaEventRecord(mstop);
			cudaEventSynchronize(mstop);
			cudaEventElapsedTime(&time_ms, mstart, mstop);

			if (!measuring && time_ms > bp.warmup_ms) {
				measuring = 1;
				measure_events = 0;
				cudaEventRecord(mstart);
			} else if (measuring && time_ms > bp.measure_ms) {
				float rate = measure_events / time_ms / 1000.0f;
				sum_rates += rate;
				n_measurements++;

				printf("  [run %d, sample %d] MEv/s: %8.3f | GVT: %.2f\n",
					r, n_measurements, rate, gvt);

				measuring = 0;
				measure_events = 0;
				cudaEventRecord(mstart);
			}

			if (measuring) {
				measure_events += h_n_events_cmt;
			}

			/* Handle next events */
			while (1) {
				h_inac_1 = h_inac_2 = h_inac_3 = 0;
				h_inac_4 = h_inac_5 = h_inac_6 = 0;
				cudaMemcpy(d_inac_1, &h_inac_1, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_2, &h_inac_2, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_3, &h_inac_3, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_4, &h_inac_4, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_5, &h_inac_5, sizeof(uint),
					cudaMemcpyHostToDevice);
				cudaMemcpy(d_inac_6, &h_inac_6, sizeof(uint),
					cudaMemcpyHostToDevice);

				total_execute_cycles++;

#if (OPTM_SYNC == 1)
				kernel_handle_next_event<<<
					n_blocks, threads_per_block>>>(
						gvt, window_size,
						d_inac_1, d_inac_2, d_inac_3,
						d_inac_4, d_inac_5, d_inac_6);
#else
				kernel_handle_next_event<<<
					n_blocks, threads_per_block>>>(
						gvt, h_lookahead,
						d_inac_1, d_inac_2, d_inac_3,
						d_inac_4, d_inac_5, d_inac_6);
#endif

				cudaMemcpy(&h_inac_1, d_inac_1, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_2, d_inac_2, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_3, d_inac_3, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_4, d_inac_4, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_5, d_inac_5, sizeof(uint),
					cudaMemcpyDeviceToHost);
				cudaMemcpy(&h_inac_6, d_inac_6, sizeof(uint),
					cudaMemcpyDeviceToHost);

#if (ALLOW_ME == 1)
				uint inac =
					h_inac_1 + h_inac_2 + h_inac_3 +
					h_inac_4 + h_inac_5 + h_inac_6;
				if (inac >= n_lps * inactive_lps_percent) { break; }
#else
				break;
#endif
			}

#if (OPTM_SYNC == 1)
			while (1) {
				h_rollback_performed = 0;
				cudaMemcpy(d_rollback_performed, &h_rollback_performed,
					sizeof(char), cudaMemcpyHostToDevice);

				kernel_roll_back<<<n_blocks, threads_per_block>>>(
					d_rollback_performed);

				cudaMemcpy(&h_rollback_performed, d_rollback_performed,
					sizeof(char), cudaMemcpyDeviceToHost);
				if (h_rollback_performed == 0) { break; }
			}
#endif

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("FATAL ERROR: %s\n", cudaGetErrorString(err));
				return 1;
			}

			kernel_sort_event_queues<<<n_blocks, threads_per_block>>>();
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_ms, start, stop);

		float avg_rate = total_events / time_ms / 1000.0f;
		float sampled_avg = n_measurements > 0 ? sum_rates / n_measurements : avg_rate;
		final_rate = sampled_avg;

		float activity_pct = total_execute_cycles > 0
			? 100.0f * total_events / ((float)total_execute_cycles * n_lps)
			: 0;

		printf("  Run %d: overall=%.3f MEv/s, sampled_avg=%.3f MEv/s, total=%.0f committed, time=%.0f ms\n",
			r, avg_rate, sampled_avg, total_events, time_ms);
		printf("  %%activity=%.2f%% (committed / (execute_cycles=%lu x n_lps=%u))\n",
			activity_pct, (unsigned long)total_execute_cycles, n_lps);

		if (bp.count_rollbacks) {
			uint h_total_processed = 0, h_total_rolledback = 0;
			cudaMemcpyFromSymbol(&h_total_processed, g_n_total_processed, sizeof(uint));
			cudaMemcpyFromSymbol(&h_total_rolledback, g_n_total_rolledback, sizeof(uint));

			float committed = total_events;
			float processed = (float)h_total_processed;
			float rolledback = (float)h_total_rolledback;
			float commit_pct = processed > 0 ? 100.0f * committed / processed : 0;
			float rollback_pct = processed > 0 ? 100.0f * rolledback / processed : 0;
			float effective_work = processed - 2.0f * rolledback;

			printf("  Stats:  processed=%10.0f | committed=%10.0f | rolledback=%10.0f\n",
				processed, committed, rolledback);
			printf("          commit%%=%.1f%% | rollback%%=%.1f%% | effective_work=%.0f\n",
				commit_pct, rollback_pct, effective_work);
			printf("          processed MEv/s=%.3f | rollback MEv/s=%.3f\n",
				processed / time_ms / 1000.0f, rolledback / time_ms / 1000.0f);
		}

		cudaEventDestroy(mstart);
		cudaEventDestroy(mstop);
	} /* end full mode */

		results_rate[r] = final_rate;

		/* Cleanup */
		free_queues();
		free_nodes();

		cudaFree(d_model_params);
		cudaFree(d_lookahead);
		cudaFree(d_ts_temp);
		cudaFree(d_n_events_cmt);
		cudaFree(d_inac_1); cudaFree(d_inac_2); cudaFree(d_inac_3);
		cudaFree(d_inac_4); cudaFree(d_inac_5); cudaFree(d_inac_6);
		cudaFree(d_rollback_performed);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	/* Summary */
	printf("\n=== Results ===\n");
	float mean_rate = get_mean(n_runs, results_rate);
	printf("Mean MEv/s: %.3f\n", mean_rate);
	if (n_runs > 1) {
		float ci = get_interval_95(n_runs, results_rate);
		printf("95%% CI:     +/- %.3f\n", ci);
	}

	return 0;
}

/* --- Utility functions (same as original) --- */

size_t get_free_memory() {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free;
}

uint get_number_blocks(uint n_threads) {
	return n_threads / threads_per_block +
		(n_threads % threads_per_block == 0 ? 0 : 1);
}

double get_gvt(double *d_ts_temp) {
	kernel_get_gvt_1<<<n_blocks, threads_per_block,
			threads_per_block * sizeof(double)>>>(d_ts_temp);

	uint next_n_blocks = get_number_blocks(n_blocks);
	uint n_left = n_blocks;
	uint distance = 1;

	while (n_left != 1) {
		kernel_get_gvt_2<<<next_n_blocks, threads_per_block,
			threads_per_block * sizeof(double)>>>(
				d_ts_temp, n_left, distance);
		n_left = next_n_blocks;
		next_n_blocks = get_number_blocks(next_n_blocks);
		distance *= threads_per_block;
	}

	double gvt;
	cudaMemcpy(&gvt, d_ts_temp, sizeof(double),
		cudaMemcpyDeviceToHost);

	return gvt;
}
