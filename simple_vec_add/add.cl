void kernel simple_add(global const int* A, global const int* B, global int* C, 
		global const int* N) {
	int ID, Nthreads, n, ratio, start, stop;

	ID = get_global_id(0);
	Nthreads = get_global_size(0);
	n = N[0];

	ratio = (n / Nthreads);  // number of elements for each thread
	start = ratio * ID;
	stop  = ratio * (ID + 1);

	for (int i=start; i<stop; i++)
		C[i] = A[i] + B[i];
}
