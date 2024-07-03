### Download AF2 parameters
cd ./src/AF2
mkdir params
cd params
wget https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar -xf alphafold_params_2021-07-14.tar
rm alphafold_params_2021-07-14.tar
cd ../../../

#Python packages
conda env create -f environment.yml
wait
conda activate evobind
#pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda deactivate

## HHblits (requires cmake)
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ../..

cd data
### Download uniclust30_2018_08
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
rm uniclust30_2018_08_hhsuite.tar.gz
