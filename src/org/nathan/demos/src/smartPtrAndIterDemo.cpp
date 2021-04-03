//
// Created by Nathan on 2021/3/15.
//

#include <iostream>
#include <memory>

namespace org::nathan::smartPtrAndIterDemo
{
    class DoubleLinedNode
    {
    public:
        int id;
        bool destroyed = false;
        std::weak_ptr<DoubleLinedNode> left;
        std::shared_ptr<DoubleLinedNode> right = nullptr;

        explicit DoubleLinedNode(int id) noexcept
        {
            this->id = id;
            std::cout << "create: " << id << std::endl;
        }


        DoubleLinedNode(const DoubleLinedNode &other)
        {
            std::cout << "Node copy " << other.id << std::endl;
            this->id = other.id;
            this->left = other.left;
            this->right = other.right;
        }

        ~DoubleLinedNode()
        {
            if (!destroyed)
            {
                std::cout << "destroy: " << id << std::endl;
                destroyed = true;
            }
            else
            {
                std::cerr << "destroy " << id << " multiple times" << std::endl;
            }
        }
    };

    struct DLIter;

    class DoubleLinkedList
    {
        // abuse utils ptr, performance issue
        std::shared_ptr<DoubleLinedNode> sentinel = std::make_shared<DoubleLinedNode>(0);
        std::shared_ptr<DoubleLinedNode> head = nullptr;
    public:
        void Add(int id)
        {
            if (head == nullptr)
            {
                head = std::make_shared<DoubleLinedNode>(id);
                head->left = sentinel;
                head->right = sentinel;
                sentinel->left = head;
                sentinel->right = head;
            }
            else
            {
                auto left = sentinel->left.lock();
                auto n = std::make_shared<DoubleLinedNode>(id);
                sentinel->left = n;
                n->right = sentinel;
                n->left = left;
                left->right = n;
            }
        }

        ~DoubleLinkedList()
        {
            sentinel->right = nullptr;
        }

        friend DLIter begin(const DoubleLinkedList &dl);

        friend DLIter end(const DoubleLinkedList &dl);
    };

    struct DLIter
    {
    private:
        std::shared_ptr<DoubleLinedNode> ptr = nullptr;
    public:
        explicit DLIter(std::shared_ptr<DoubleLinedNode> n)
        {
            ptr = std::move(n);
        }

        DLIter(DLIter &other)
        {
            std::cout << "DLIter copy" << std::endl;
            this->ptr = other.ptr;
        }

        DoubleLinedNode &operator*() const
        {
            return *ptr;
        }

        bool operator!=(const DLIter &other) const
        {
            return other.ptr != ptr;
        }

        bool operator==(const DLIter &other) const
        {
            std::cout << "no return, no crash\n";
            return this == &other;
        }

        void operator++()
        {
            ptr = ptr->right;
        }
    };


    inline DLIter begin(const DoubleLinkedList &dl)
    {
        return DLIter(dl.head);
    }

    inline DLIter end(const DoubleLinkedList &dl)
    {
        return DLIter(dl.sentinel);
    }

    [[maybe_unused]] void destructionAndForRangeDemo()
    {
        {
            std::unique_ptr<DoubleLinkedList> dl = std::make_unique<DoubleLinkedList>();
            dl->Add(3);
            dl->Add(17);
            dl->Add(24);
            dl->Add(23);
            dl->Add(7);
            auto t = *dl;
            for (auto &item : t)
            {
                std::cout << "iterating: " << item.id << std::endl;
                item.id++;
            }
            std::cout << "iterate end" << std::endl;
            for (auto &item : *dl)
            {
                std::cout << "iterating: " << item.id << std::endl;
            }
        }
        std::cout << "destroy end\n";
        {
            auto v{std::make_shared<DoubleLinedNode>(2)};
            {
                auto t = *(v.get());
                std::cout << t.id << std::endl;
                t.id = 3;
            }
            std::cout << "destruct copied object before" << std::endl;
        }
    }
}
