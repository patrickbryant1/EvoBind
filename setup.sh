### Download AF2 parameters
cd ./src/AF2
mkdir params
cd params
wget https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar -xf alphafold_params_2021-07-14.tar
rm alphafold_params_2021-07-14.tar

### Create singularity environment to run AF2
cd ..
singularity pull AF_environment.sif docker://catgumag/alphafold:latest

cd ../../data/
### Download uniclust30_2018_08
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
rm uniclust30_2018_08_hhsuite.tar.gz
