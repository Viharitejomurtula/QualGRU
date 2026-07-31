#include <iostream>

#include "gru_cell.hpp"



int main() {

    QualGRU model("export_h256", 50, 256, 32, 8);



    int hidden_size = model.cell.hidden_size;

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim = qual_emb_dim + 2 * base_emb_dim;


    std::vector<std::vector<int>> qual_ids = {{6, 10, 20, 5}, {6, 15}};
    std::vector<std::vector<int>> base_ids  = {{0, 1, 2, 3},  {0, 1}};

    int batch_size = 2;
    int max_len = 4;
    //old
    std::vector<Eigen::VectorXf> reference_h(batch_size, Eigen::VectorXf::Zero(hidden_size));
    GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

    for (int r = 0; r < batch_size; r++) {
        Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_size);
        int len = qual_ids[r].size();
        for (int t = 0; t < len - 1; t++) {
            model.cell.forward_into(qual_ids[r][t], base_ids[r][t], base_ids[r][t+1], h, buf);
            h = buf.h_new;
        }
        reference_h[r] = h;
        std::cout << "Reference read " << r << " final h, first 5: " << h.head(5).transpose() << std::endl;
    }
    BatchGRUBuffers bbuf(hidden_size, input_dim, qual_emb_dim, base_emb_dim, batch_size);

    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(hidden_size, batch_size);
    for (int t = 0; t < max_len - 1; t++) {
        Eigen::RowVectorXf mask(batch_size);
        std::vector<int> q_idx(batch_size), b_idx(batch_size), bn_idx(batch_size);

        for (int r = 0; r < batch_size; r++) {
            int len = qual_ids[r].size();
            bool active = (t < len - 1);
            mask(r) = active ? 1.0f : 0.0f;

            // For finished reads, just repeat their last valid index (value doesn't matter, mask zeroes it out)
            int tt = active ? t : (len - 2);
            q_idx[r] = qual_ids[r][tt];
            b_idx[r] = base_ids[r][tt];
            bn_idx[r] = base_ids[r][tt + 1];
        }

        model.cell.forward_batch(q_idx, b_idx, bn_idx, H, mask, bbuf);
        H = bbuf.H_new;
    }

    for (int r = 0; r <batch_size; r++) {
        std::cout << "Batch read" << r << "  final h, first 5: " << H.col(r).head(5).transpose() << std::endl;
    }



    Eigen::VectorXf probs_ref = model.predict_probs(reference_h[0]);
    Eigen::MatrixXf H_final(hidden_size, batch_size);
    H_final.col(0) = reference_h[0];
    H_final.col(1) = reference_h[1];
    Eigen::MatrixXf probs_batch = model.predict_probs_batch(H_final);

    std::cout << "Reference probs[0], first 5: " << probs_ref.head(5).transpose() << std::endl;
    std::cout << "Batched probs col 0, first 5: " << probs_batch.col(0).head(5).transpose() << std::endl;

    return 0;

}
