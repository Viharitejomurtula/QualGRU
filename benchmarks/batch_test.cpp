#include <iostream>
#include <Eigen/Dense>

int main() {
	int qual_vocab_size = 5;
	int qual_emb_dim = 3;
	int base_emb_dim = 2;
	int batch_size = 4;

	Eigen::MatrixXf qual_emb(qual_vocab_size, qual_emb_dim);
	qual_emb << 1, 2, 3,
		    4, 5, 6,
		    7, 8, 9,
		   10, 11, 12,
		   13, 14, 15;


	Eigen::MatrixXf base_emb(5, base_emb_dim);
    	base_emb << 0.1, 0.2,
        	    0.3, 0.4,
              	    0.5, 0.6,
                    0.7, 0.8,
                    0.9, 1.0;

	std::vector<int> qual_indices = {0,2,1,4};
	std::vector<int> base_indices =  {0,1,2,3};
	std::vector<int> base_next_indices {1,2,3,0};


	int input_dim = qual_emb_dim + 2 * base_emb_dim;
	Eigen::MatrixXf X_t(input_dim, batch_size);

	int hidden_size = 6;
	Eigen::MatrixXf W_ru = Eigen::MatrixXf::Random(2 * hidden_size, hidden_size);
        Eigen::MatrixXf U_ru = Eigen::MatrixXf::Random(2 * hidden_size, input_dim);
        Eigen::VectorXf b_ru = Eigen::VectorXf::Random(2 * hidden_size);
	Eigen::MatrixXf H_prev = Eigen::MatrixXf::Zero(hidden_size, batch_size);

	Eigen::MatrixXf RU_pre = W_ru * H_prev + U_ru * X_t;
	RU_pre.colwise() += b_ru;

	std::cout << "\nRU_pre shape: " << RU_pre.rows() << " x " << RU_pre.cols() << std::endl;

	for (int i=0; i < batch_size; i++) {
		X_t.block(0,i,qual_emb_dim,1) = qual_emb.row(qual_indices[i]).transpose();
		X_t.block(qual_emb_dim, i, base_emb_dim, 1) = base_emb.row(base_indices[i]).transpose();
		X_t.block(qual_emb_dim + base_emb_dim, i, base_emb_dim, 1) = base_emb.row(base_next_indices[i]).transpose();


	}
    	std::cout << "X_t (each column = one read's full concatenated input):" << std::endl;
    	std::cout << X_t << std::endl;

	std::vector<int> read_lengths = {5, 3, 7, 2};
	int t= 2;

	Eigen::RowVectorXf mask(batch_size);
	for (int i=0; i < batch_size; i++) {
		mask(i) = (t < read_lengths[i] - 1) ? 1.0f : 0.0f;
	}
	std::cout << "\nMask at t=" << t << ": " << mask << std::endl;


	Eigen::MatrixXf H_update = Eigen::MatrixXf::Random(hidden_size, batch_size);
	Eigen::MatrixXf H_new(hidden_size, batch_size);

	for (int i=0; i < batch_size; i++) {
		H_new.col(i) = mask(i) * H_update.col(i) + (1.0f - mask(i)) * H_prev.col(i);
	}
	std::cout << "H_new shape: " << H_new.rows() << " x " << H_new.cols() << std::endl;




	return 0;

}
