make par_top_pmd -j
mpirun -np 8 ./par_top_pmd -rs 2 -rp 3 -p 1 -fr 0.1 -pos-bregman -bregman
mpirun -np 8 -./par_top_pmd rs 2 -rp 4 -p 1 -fr 0.1 -pos-bregman -bregman
mpirun -np 8 ./par_top_pmd -rs 2 -rp 5 -p 1 -fr 0.1 -pos-bregman -bregman
mpirun -np 8 ./par_top_pmd -rs 2 -rp 4 -p -2 -fr -pos-bregman -bregman
