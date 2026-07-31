#ifndef GRU_CELL_HPP

#define GRU_CELL_HPP



#include <Eigen/Dense>

#include <fstream>

#include <vector>

#include <string>

#include <stdexcept>

#include <iostream>





struct BatchGRUBuffers {

        Eigen::MatrixXf QE, BE_0, BE_1, X_t;

        Eigen::MatrixXf RU_pre, R, U, C_pre, C, H_new;



        BatchGRUBuffers(int hidden_size, int input_dim, int qual_emb_dim, int base_emb_dim, int batch_size) {

                QE.resize(qual_emb_dim, batch_size);

                BE_0.resize(base_emb_dim, batch_size);

                BE_1.resize(base_emb_dim, batch_size);

                X_t.resize(input_dim, batch_size);

                RU_pre.resize(2 * hidden_size, batch_size);

                R.resize(hidden_size, batch_size);

                U.resize(hidden_size, batch_size);

                C_pre.resize(hidden_size, batch_size);

                C.resize(hidden_size, batch_size);

                H_new.resize(hidden_size, batch_size);

        }

};





Eigen::MatrixXf load_matrix(const std::string& path, int rows, int cols){

        std::ifstream file(path, std::ios::binary);

        if (!file) {

                throw std::runtime_error("Couldn't open file: " + path);

        }

        std::vector<float> buffer(rows * cols);

        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));



        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped(buffer.data(), rows, cols);

        Eigen::MatrixXf result = mapped;

        return result;

}



Eigen::VectorXf load_vector(const std::string& path, int size){

        std::ifstream file(path, std::ios::binary);

        if (!file) {

                throw std::runtime_error("Couldn't open file: " + path);

        }



        Eigen::VectorXf result(size);

        file.read(reinterpret_cast<char*>(result.data()), size * sizeof(float));

        return result;

}





// Reusable per-thread scratch buffers, so forward_into() never allocates

// new memory on the heap during the hot per-timestep loop.

struct GRUBuffers {

        Eigen::VectorXf qe, be_0, be_1, x_t;

        Eigen::VectorXf r_pre, r, u_pre, u, c_pre, c, h_new;

        Eigen::VectorXf ru_pre;  // fused r_pre/u_pre result, length 2*hidden_size



        GRUBuffers(int hidden_size, int input_dim, int qual_emb_dim, int base_emb_dim) {

                qe.resize(qual_emb_dim);

                be_0.resize(base_emb_dim);

                be_1.resize(base_emb_dim);

                x_t.resize(input_dim);

                r_pre.resize(hidden_size);

                r.resize(hidden_size);

                u_pre.resize(hidden_size);

                u.resize(hidden_size);

                c_pre.resize(hidden_size);

                c.resize(hidden_size);

                h_new.resize(hidden_size);

                ru_pre.resize(2 * hidden_size);

        }

};





class BaseConditionedGRUCell {

public:

        int hidden_size;

        int qual_emb_dim;

        int base_emb_dim;



        Eigen::MatrixXf qual_emb;

        Eigen::MatrixXf base_emb;

        Eigen::MatrixXf U_r, U_c, U_u;

        Eigen::MatrixXf W_r, W_c, W_u;

        Eigen::VectorXf b_r, b_c, b_u;



        Eigen::MatrixXf W_ru;  // (2*hidden_size) x hidden_size

        Eigen::MatrixXf U_ru;  // (2*hidden_size) x input_dim

        Eigen::VectorXf b_ru;  // (2*hidden_size)





        BaseConditionedGRUCell(int qual_vocab_size, int hidden_size_, int qual_emb_dim_, int base_emb_dim_){

                hidden_size = hidden_size_;

                qual_emb_dim = qual_emb_dim_;

                base_emb_dim = base_emb_dim_;



                int input_dims = qual_emb_dim + 2 * base_emb_dim;



                qual_emb.resize(qual_vocab_size, qual_emb_dim);

                base_emb.resize(5, base_emb_dim);



                U_r.resize(hidden_size, input_dims);

                U_u.resize(hidden_size, input_dims);

                U_c.resize(hidden_size, input_dims);



                W_r.resize(hidden_size, hidden_size);

                W_u.resize(hidden_size, hidden_size);

                W_c.resize(hidden_size, hidden_size);



                b_r.resize(hidden_size);

                b_u.resize(hidden_size);

                b_c.resize(hidden_size);

        }



        BaseConditionedGRUCell(const std::string& weights_dir, int qual_vocab_size, int hidden_size_, int qual_emb_dim_, int base_emb_dim_) {

                hidden_size = hidden_size_;

                qual_emb_dim = qual_emb_dim_;

                base_emb_dim = base_emb_dim_;

                int input_dims = qual_emb_dim + 2 * base_emb_dim;



                qual_emb = load_matrix(weights_dir + "/cell_qual_emb_weight.bin", qual_vocab_size, qual_emb_dim);

                base_emb = load_matrix(weights_dir + "/cell_base_emb_weight.bin", 5, base_emb_dim);



                U_r = load_matrix(weights_dir + "/cell_U_r_weight.bin", hidden_size, input_dims);

                U_u = load_matrix(weights_dir + "/cell_U_u_weight.bin", hidden_size, input_dims);

                U_c = load_matrix(weights_dir + "/cell_U_c_weight.bin", hidden_size, input_dims);



                W_r = load_matrix(weights_dir + "/cell_W_r_weight.bin", hidden_size, hidden_size);

                W_u = load_matrix(weights_dir + "/cell_W_u_weight.bin", hidden_size, hidden_size);

                W_c = load_matrix(weights_dir + "/cell_W_c_weight.bin", hidden_size, hidden_size);



                b_u = load_vector(weights_dir + "/cell_W_u_bias.bin", hidden_size);

                b_r = load_vector(weights_dir + "/cell_W_r_bias.bin", hidden_size);

                b_c = load_vector(weights_dir + "/cell_W_c_bias.bin", hidden_size);



                W_ru.resize(2 * hidden_size, hidden_size);

                W_ru.topRows(hidden_size) = W_r;

                W_ru.bottomRows(hidden_size) = W_u;



                U_ru.resize(2 * hidden_size, input_dims);

                U_ru.topRows(hidden_size) = U_r;

                U_ru.bottomRows(hidden_size) = U_u;



                b_ru.resize(2 * hidden_size);

                b_ru.head(hidden_size) = b_r;

                b_ru.tail(hidden_size) = b_u;

        }



        Eigen::VectorXf forward(int qual_idx, int base_idx, int base_next_idx, const Eigen::VectorXf& h_prev){

                Eigen::VectorXf qe = qual_emb.row(qual_idx).transpose();

                Eigen::VectorXf be_0 = base_emb.row(base_idx).transpose();

                Eigen::VectorXf be_1 = base_emb.row(base_next_idx).transpose();



                int input_dims = qual_emb_dim + 2 * base_emb_dim;

                Eigen::VectorXf x_t(input_dims);

                x_t.segment(0, qual_emb_dim) = qe;

                x_t.segment(qual_emb_dim, base_emb_dim) = be_0;

                x_t.segment(qual_emb_dim + base_emb_dim, base_emb_dim) = be_1;



                Eigen::VectorXf r_pre = W_r * h_prev + b_r + U_r * x_t;

                Eigen::VectorXf r = (1.0f / (1.0f + (-r_pre.array()).exp())).matrix();



                Eigen::VectorXf u_pre = W_u * h_prev + b_u + U_u * x_t;

                Eigen::VectorXf u = (1.0f / (1.0f + (-u_pre.array()).exp())).matrix();



                Eigen::VectorXf c_pre = W_c * (r.cwiseProduct(h_prev)) + b_c + U_c * x_t;

                Eigen::VectorXf c = c_pre.array().tanh().matrix();



                Eigen::VectorXf h_new = (1.0f - u.array()).matrix().cwiseProduct(h_prev) + u.cwiseProduct(c);



                return h_new;

        }



        void forward_into(int qual_idx, int base_idx, int base_next_idx, const Eigen::VectorXf& h_prev, GRUBuffers& buf) {

                buf.qe = qual_emb.row(qual_idx).transpose();

                buf.be_0 = base_emb.row(base_idx).transpose();

                buf.be_1 = base_emb.row(base_next_idx).transpose();



                buf.x_t.segment(0, qual_emb_dim) = buf.qe;

                buf.x_t.segment(qual_emb_dim, base_emb_dim) = buf.be_0;

                buf.x_t.segment(qual_emb_dim + base_emb_dim, base_emb_dim) = buf.be_1;



                buf.ru_pre.noalias() = W_ru * h_prev + b_ru + U_ru * buf.x_t;



                buf.r = (1.0f / (1.0f + (-buf.ru_pre.head(hidden_size).array()).exp())).matrix();

                buf.u = (1.0f / (1.0f + (-buf.ru_pre.tail(hidden_size).array()).exp())).matrix();



                buf.c_pre.noalias() = W_c * (buf.r.cwiseProduct(h_prev)) + b_c + U_c * buf.x_t;

                buf.c = buf.c_pre.array().tanh().matrix();



                buf.h_new = (1.0f - buf.u.array()).matrix().cwiseProduct(h_prev) + buf.u.cwiseProduct(buf.c);

        }



        // Batched version: processes an entire batch of reads for ONE

        // timestep in a single call. H_prev is hidden_size x batch_size

        // (one column per read). mask has one entry per read: 1.0 if that

        // read is still active at this timestep, 0.0 if it already

        // finished (frozen, keeps its old H_prev value unchanged).

        void forward_batch(const std::vector<int>& qual_idx,

                            const std::vector<int>& base_idx,

                            const std::vector<int>& base_next_idx,

                            const Eigen::MatrixXf& H_prev,

                            const Eigen::RowVectorXf& mask,

                            BatchGRUBuffers& buf) {

                int batch_size = H_prev.cols();



                for (int i = 0; i < batch_size; i++) {

                        buf.QE.col(i) = qual_emb.row(qual_idx[i]).transpose();

                        buf.BE_0.col(i) = base_emb.row(base_idx[i]).transpose();

                        buf.BE_1.col(i) = base_emb.row(base_next_idx[i]).transpose();

                }



                buf.X_t.block(0, 0, qual_emb_dim, batch_size) = buf.QE;

                buf.X_t.block(qual_emb_dim, 0, base_emb_dim, batch_size) = buf.BE_0;

                buf.X_t.block(qual_emb_dim + base_emb_dim, 0, base_emb_dim, batch_size) = buf.BE_1;



                buf.RU_pre.noalias() = W_ru * H_prev + U_ru * buf.X_t;

                buf.RU_pre.colwise() += b_ru;



                buf.R = (1.0f / (1.0f + (-buf.RU_pre.topRows(hidden_size).array()).exp())).matrix();

                buf.U = (1.0f / (1.0f + (-buf.RU_pre.bottomRows(hidden_size).array()).exp())).matrix();



                buf.C_pre.noalias() = W_c * (buf.R.cwiseProduct(H_prev)) + U_c * buf.X_t;

                buf.C_pre.colwise() += b_c;

                buf.C = buf.C_pre.array().tanh().matrix();



                Eigen::MatrixXf H_update = (1.0f - buf.U.array()).matrix().cwiseProduct(H_prev) + buf.U.cwiseProduct(buf.C);



                for (int i = 0; i < batch_size; i++) {

                        buf.H_new.col(i) = mask(i) * H_update.col(i) + (1.0f - mask(i)) * H_prev.col(i);

                }

        }


	// Same as forward_batch, but X_t is precomputed and passed in

        // directly -- no embedding lookups happen here. Use this when

        // you've already built all timesteps' X_t matrices upfront.

        void forward_batch_precomputed(const Eigen::MatrixXf& X_t,

                                        const Eigen::MatrixXf& H_prev,

                                        const Eigen::RowVectorXf& mask,

                                        BatchGRUBuffers& buf) {

                buf.RU_pre.noalias() = W_ru * H_prev + U_ru * X_t;

                buf.RU_pre.colwise() += b_ru;



                buf.R = (1.0f / (1.0f + (-buf.RU_pre.topRows(hidden_size).array()).exp())).matrix();

                buf.U = (1.0f / (1.0f + (-buf.RU_pre.bottomRows(hidden_size).array()).exp())).matrix();



                buf.C_pre.noalias() = W_c * (buf.R.cwiseProduct(H_prev)) + U_c * X_t;

                buf.C_pre.colwise() += b_c;

                buf.C = buf.C_pre.array().tanh().matrix();



                int batch_size = H_prev.cols();

                Eigen::MatrixXf H_update = (1.0f - buf.U.array()).matrix().cwiseProduct(H_prev) + buf.U.cwiseProduct(buf.C);

                for (int i = 0; i < batch_size; i++) {

                        buf.H_new.col(i) = mask(i) * H_update.col(i) + (1.0f - mask(i)) * H_prev.col(i);

                }

        }

};





class QualGRU {

public:

        BaseConditionedGRUCell cell;

        Eigen::MatrixXf out_proj_weight;

        Eigen::VectorXf out_proj_bias;



        QualGRU(const std::string& weights_dir, int qual_vocab_size, int hidden_size, int qual_emb_dim, int base_emb_dim) : cell(weights_dir, qual_vocab_size, hidden_size, qual_emb_dim, base_emb_dim) {

                out_proj_weight = load_matrix(weights_dir + "/out_proj_weight.bin", qual_vocab_size, hidden_size);

                out_proj_bias = load_vector(weights_dir + "/out_proj_bias.bin", qual_vocab_size);

        }



        Eigen::VectorXf predict_probs(const Eigen::VectorXf& h) {

                Eigen::VectorXf logits = out_proj_weight * h + out_proj_bias;

                Eigen::VectorXf shifted = logits.array() - logits.maxCoeff();

                Eigen::VectorXf exp_vals = shifted.array().exp();

                float sum_exp = exp_vals.sum();

                return exp_vals / sum_exp;

        }


	Eigen::MatrixXf predict_probs_batch(const Eigen::MatrixXf& H) {

        Eigen::MatrixXf logits = out_proj_weight * H;

        logits.colwise() += out_proj_bias;



        Eigen::RowVectorXf col_max = logits.colwise().maxCoeff();

        Eigen::MatrixXf shifted = logits.rowwise() - col_max;

        Eigen::MatrixXf exp_vals = shifted.array().exp();

        Eigen::RowVectorXf col_sum = exp_vals.colwise().sum();

        Eigen::MatrixXf probs = exp_vals.array().rowwise() / col_sum.array();



        return probs;

}
};



#endif
