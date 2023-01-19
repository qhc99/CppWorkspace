//
// Created by Nathan on 2023-01-17.
//

#ifndef DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP
#define DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP

#include <string>
#include "lib_central/utils.h"

using std::string;
using dev::qhc::utils::position_in_file;

template<typename T_Val_Copyable>
class TernaryTries final {
public:
    class Node;

private:
    Node *root{nullptr};
    int count{};

    bool get(Node *n, const string &key, int depth, T_Val_Copyable &ret) {
        if (n == nullptr) {
            return false;
        }
        char c = key.at(depth);
        if (c < n->chr) { return get(n->left, key, depth); }
        else if (c > n->chr) { return get(n->right, key, depth); }
        else if (depth < key.length() - 1) { return get(n->mid, key, depth + 1); }
        else {
            ret = n->val;
            return true;
        }
    }

    void recursive_release(Node *n) noexcept {
        if (n->left != nullptr) {
            recursive_release(n->left);
        }
        if (n->mid != nullptr) {
            recursive_release(n->mid);
        }
        if (n->right != nullptr) {
            recursive_release(n->right);
        }
        delete n;
    }

public:

    class Node {
    private:
        friend TernaryTries;
        char chr{};
        bool contain{false};
        Node *left{nullptr}, *mid{nullptr}, *right{nullptr};
        T_Val_Copyable val{};

        Node() = default;

        ~Node() = default;
    public:

        [[nodiscard]] char getNodeChar() const {
            return chr;
        }

        [[nodiscard]] bool containValue() const {
            return contain;
        }

        /**
         *
         * @return return copy or return pointer
         */
        T_Val_Copyable getValue() const {
            return val;
        }

        const Node *getLeft() const {
            return left;
        }

        const Node *getMid() const {
            return mid;
        }

        const Node *getRight() const {
            return right;
        }

        Node(const Node &other) = delete;

        Node(Node &&other) = delete;

        Node &operator=(const Node &other) = delete;

        Node &operator=(Node &&other) = delete;


    };

    TernaryTries() = default;

    TernaryTries(const TernaryTries<T_Val_Copyable> &other) = delete;

    TernaryTries(TernaryTries<T_Val_Copyable> &&other) noexcept = default;

    TernaryTries &operator=(const TernaryTries &other) = delete;

    TernaryTries &operator=(TernaryTries &&other) noexcept = default;


    ~TernaryTries() {
        recursive_release(root);
    }

    const Node *getRoot() {
        return root;
    }

    /**
     *
     * @return count of inserted key value pair
     */
    int getCount() {
        return count;
    }

    /**
     *
     * @param key string eky
     * @param ret value
     * @return if key exists
     */
    bool getValueOfKey(const string &key, T_Val_Copyable &ret) {
        if (key.empty()) {
            return false;
        }
        return get(root, key, ret);
    }
};

#endif //DEV_QHC_CPP_PROJECTS_TERNARYTRIES_HPP
