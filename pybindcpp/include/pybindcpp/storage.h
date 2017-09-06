// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef STORAGE_H
#define STORAGE_H

#include <list>
#include <memory>

// Used to store static variables in dynamically loaded module
// Should we add it as a Python Object to module instead of static?
static std::list<std::shared_ptr<void>> __storage__;

template <class T>
T*
store(T&& t)
{
  auto p = std::make_shared<T>(std::forward<T>(t));
  auto v = std::static_pointer_cast<void>(p);
  __storage__.push_back(v);
  return p.get();
}

template <class T>
T*
store(const T& t)
{
  auto p = std::make_shared<T>(t);
  auto v = std::static_pointer_cast<void>(p);
  __storage__.push_back(v);
  return p.get();
}

#endif // STORAGE_H
