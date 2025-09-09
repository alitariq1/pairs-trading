from typing import overload
from enum import Enum
import abc
import typing
import warnings

import System
import System.Collections
import System.Globalization
import System.Runtime.Serialization


class IEnumerator(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    @property
    @abc.abstractmethod
    def current(self) -> System.Object:
        ...

    def move_next(self) -> bool:
        ...

    def reset(self) -> None:
        ...


class IEnumerable(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    def get_enumerator(self) -> System.Collections.IEnumerator:
        ...


class ICollection(System.Collections.IEnumerable, metaclass=abc.ABCMeta):
    """This class has no documentation."""

    @property
    @abc.abstractmethod
    def count(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def sync_root(self) -> System.Object:
        ...

    @property
    @abc.abstractmethod
    def is_synchronized(self) -> bool:
        ...

    def copy_to(self, array: System.Array, index: int) -> None:
        ...


class IDictionary(System.Collections.ICollection, metaclass=abc.ABCMeta):
    """This class has no documentation."""

    @property
    @abc.abstractmethod
    def keys(self) -> System.Collections.ICollection:
        ...

    @property
    @abc.abstractmethod
    def values(self) -> System.Collections.ICollection:
        ...

    @property
    @abc.abstractmethod
    def is_read_only(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def is_fixed_size(self) -> bool:
        ...

    def __getitem__(self, key: typing.Any) -> typing.Any:
        ...

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def add(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def clear(self) -> None:
        ...

    def contains(self, key: typing.Any) -> bool:
        ...

    def remove(self, key: typing.Any) -> None:
        ...


class IHashCodeProvider(metaclass=abc.ABCMeta):
    """
    Provides a mechanism for a Hashtable user to override the default
    GetHashCode() function on Objects, providing their own hash function.
    
    IHashCodeProvider has been deprecated. Use IEqualityComparer instead.
    """

    def get_hash_code(self, obj: typing.Any) -> int:
        """Returns a hash code for the given object."""
        ...


class IComparer(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    def compare(self, x: typing.Any, y: typing.Any) -> int:
        ...


class IEqualityComparer(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    def equals(self, x: typing.Any, y: typing.Any) -> bool:
        ...

    def get_hash_code(self, obj: typing.Any) -> int:
        ...


class DictionaryEntry:
    """This class has no documentation."""

    @property
    def key(self) -> System.Object:
        ...

    @key.setter
    def key(self, value: System.Object) -> None:
        ...

    @property
    def value(self) -> System.Object:
        ...

    @value.setter
    def value(self, value: System.Object) -> None:
        ...

    def __init__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def deconstruct(self, key: typing.Optional[typing.Any], value: typing.Optional[typing.Any]) -> typing.Tuple[None, typing.Any, typing.Any]:
        ...

    def to_string(self) -> str:
        ...


class IDictionaryEnumerator(System.Collections.IEnumerator, metaclass=abc.ABCMeta):
    """This class has no documentation."""

    @property
    @abc.abstractmethod
    def key(self) -> System.Object:
        ...

    @property
    @abc.abstractmethod
    def value(self) -> System.Object:
        ...

    @property
    @abc.abstractmethod
    def entry(self) -> System.Collections.DictionaryEntry:
        ...


class Hashtable(System.Object, System.Collections.IDictionary, System.Runtime.Serialization.ISerializable, System.Runtime.Serialization.IDeserializationCallback, System.ICloneable):
    """This class has no documentation."""

    @property
    def hcp(self) -> System.Collections.IHashCodeProvider:
        """
        This property is protected.
        
        Hashtable.hcp has been deprecated. Use the EqualityComparer property instead.
        """
        warnings.warn("Hashtable.hcp has been deprecated. Use the EqualityComparer property instead.", DeprecationWarning)

    @hcp.setter
    def hcp(self, value: System.Collections.IHashCodeProvider) -> None:
        warnings.warn("Hashtable.hcp has been deprecated. Use the EqualityComparer property instead.", DeprecationWarning)

    @property
    def comparer(self) -> System.Collections.IComparer:
        """
        This property is protected.
        
        Hashtable.comparer has been deprecated. Use the KeyComparer properties instead.
        """
        warnings.warn("Hashtable.comparer has been deprecated. Use the KeyComparer properties instead.", DeprecationWarning)

    @comparer.setter
    def comparer(self, value: System.Collections.IComparer) -> None:
        warnings.warn("Hashtable.comparer has been deprecated. Use the KeyComparer properties instead.", DeprecationWarning)

    @property
    def equality_comparer(self) -> System.Collections.IEqualityComparer:
        """This property is protected."""
        ...

    @property
    def is_read_only(self) -> bool:
        ...

    @property
    def is_fixed_size(self) -> bool:
        ...

    @property
    def is_synchronized(self) -> bool:
        ...

    @property
    def keys(self) -> System.Collections.ICollection:
        ...

    @property
    def values(self) -> System.Collections.ICollection:
        ...

    @property
    def sync_root(self) -> System.Object:
        ...

    @property
    def count(self) -> int:
        ...

    def __contains__(self, key: typing.Any) -> bool:
        ...

    def __getitem__(self, key: typing.Any) -> typing.Any:
        ...

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, capacity: int) -> None:
        ...

    @overload
    def __init__(self, capacity: int, load_factor: float) -> None:
        ...

    @overload
    def __init__(self, capacity: int, load_factor: float, equality_comparer: System.Collections.IEqualityComparer) -> None:
        ...

    @overload
    def __init__(self, equality_comparer: System.Collections.IEqualityComparer) -> None:
        ...

    @overload
    def __init__(self, capacity: int, equality_comparer: System.Collections.IEqualityComparer) -> None:
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary) -> None:
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary, load_factor: float) -> None:
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary, equality_comparer: System.Collections.IEqualityComparer) -> None:
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary, load_factor: float, equality_comparer: System.Collections.IEqualityComparer) -> None:
        ...

    @overload
    def __init__(self, hcp: System.Collections.IHashCodeProvider, comparer: System.Collections.IComparer) -> None:
        """This constructor has been deprecated. Use Hashtable(IEqualityComparer) instead."""
        ...

    @overload
    def __init__(self, capacity: int, hcp: System.Collections.IHashCodeProvider, comparer: System.Collections.IComparer) -> None:
        """This constructor has been deprecated. Use Hashtable(int, IEqualityComparer) instead."""
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary, hcp: System.Collections.IHashCodeProvider, comparer: System.Collections.IComparer) -> None:
        """This constructor has been deprecated. Use Hashtable(IDictionary, IEqualityComparer) instead."""
        ...

    @overload
    def __init__(self, capacity: int, load_factor: float, hcp: System.Collections.IHashCodeProvider, comparer: System.Collections.IComparer) -> None:
        """This constructor has been deprecated. Use Hashtable(int, float, IEqualityComparer) instead."""
        ...

    @overload
    def __init__(self, d: System.Collections.IDictionary, load_factor: float, hcp: System.Collections.IHashCodeProvider, comparer: System.Collections.IComparer) -> None:
        """This constructor has been deprecated. Use Hashtable(IDictionary, float, IEqualityComparer) instead."""
        ...

    @overload
    def __init__(self, info: System.Runtime.Serialization.SerializationInfo, context: System.Runtime.Serialization.StreamingContext) -> None:
        """
        This method is protected.
        
        Obsoletions.LegacyFormatterImplMessage
        """
        ...

    def __len__(self) -> int:
        ...

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def add(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def clear(self) -> None:
        ...

    def clone(self) -> System.Object:
        ...

    def contains(self, key: typing.Any) -> bool:
        ...

    def contains_key(self, key: typing.Any) -> bool:
        ...

    def contains_value(self, value: typing.Any) -> bool:
        ...

    def copy_to(self, array: System.Array, array_index: int) -> None:
        ...

    def get_enumerator(self) -> System.Collections.IDictionaryEnumerator:
        ...

    def get_hash(self, key: typing.Any) -> int:
        """This method is protected."""
        ...

    def get_object_data(self, info: System.Runtime.Serialization.SerializationInfo, context: System.Runtime.Serialization.StreamingContext) -> None:
        """Obsoletions.LegacyFormatterImplMessage"""
        warnings.warn("Obsoletions.LegacyFormatterImplMessage", DeprecationWarning)

    def key_equals(self, item: typing.Any, key: typing.Any) -> bool:
        """This method is protected."""
        ...

    def on_deserialization(self, sender: typing.Any) -> None:
        ...

    def remove(self, key: typing.Any) -> None:
        ...

    @staticmethod
    def synchronized(table: System.Collections.Hashtable) -> System.Collections.Hashtable:
        ...


class BitArray(System.Object, System.Collections.ICollection, System.ICloneable, System.Runtime.Serialization.ISerializable):
    """
    Manages a compact array of bit values, which are represented as bool, where
    true indicates that the bit is on (1) and false indicates
    the bit is off (0).
    """

    @property
    def length(self) -> int:
        """Gets or sets the number of elements in the BitArray."""
        ...

    @length.setter
    def length(self, value: int) -> None:
        ...

    @property
    def count(self) -> int:
        """Gets the number of elements contained in the BitArray."""
        ...

    @property
    def sync_root(self) -> System.Object:
        """Gets an object that can be used to synchronize access to the BitArray."""
        ...

    @property
    def is_synchronized(self) -> bool:
        """Gets a value indicating whether access to the BitArray is synchronized (thread safe)."""
        ...

    @property
    def is_read_only(self) -> bool:
        """Gets a value indicating whether the BitArray is read-only."""
        ...

    def __getitem__(self, index: int) -> bool:
        """
        Gets or sets the value of the bit at a specific position in the BitArray.
        
        :param index: The zero-based index of the value to get or set.
        :returns: The value of the bit at position .
        """
        ...

    @overload
    def __init__(self, length: int) -> None:
        """
        Initializes a new instance of the BitArray class that can hold the specified
        number of bit values, which are initially set to false.
        
        :param length: The number of bit values in the new BitArray.
        """
        ...

    @overload
    def __init__(self, length: int, default_value: bool) -> None:
        """
        Initializes a new instance of the BitArray class that can hold the specified number of
        bit values, which are initially set to the specified value.
        
        :param length: The number of bit values in the new BitArray.
        :param default_value: The Boolean value to assign to each bit.
        """
        ...

    @overload
    def __init__(self, bytes: typing.List[int]) -> None:
        """
        Initializes a new instance of the BitArray class that contains bit values copied
        from the specified array of bytes.
        
        :param bytes: An array of bytes containing the values to copy, where each byte represents eight consecutive bits.
        """
        ...

    @overload
    def __init__(self, values: typing.List[bool]) -> None:
        """
        Initializes a new instance of the BitArray class that contains bit values
        copied from the specified array of Booleans.
        
        :param values: An array of Booleans to copy.
        """
        ...

    @overload
    def __init__(self, values: typing.List[int]) -> None:
        """
        Initializes a new instance of the BitArray class that contains bit values
        copied from the specified array of 32-bit integers.
        
        :param values: An array of integers containing the values to copy, where each integer represents 32 consecutive bits.
        """
        ...

    @overload
    def __init__(self, bits: System.Collections.BitArray) -> None:
        """
        Initializes a new instance of the BitArray class that contains bit values copied from the specified BitArray.
        
        :param bits: The BitArray to copy.
        """
        ...

    def __setitem__(self, index: int, value: bool) -> None:
        """
        Gets or sets the value of the bit at a specific position in the BitArray.
        
        :param index: The zero-based index of the value to get or set.
        :returns: The value of the bit at position .
        """
        ...

    def And(self, value: System.Collections.BitArray) -> System.Collections.BitArray:
        """
        Performs the bitwise AND operation between the elements of the current BitArray object and the
        corresponding elements in the specified array. The current BitArray object will be modified to
        store the result of the bitwise AND operation.
        
        :param value: The array with which to perform the bitwise AND operation.
        :returns: An array containing the result of the bitwise AND operation, which is a reference to the current BitArray object.
        """
        ...

    def clone(self) -> System.Object:
        """Creates a shallow copy of the BitArray."""
        ...

    def copy_to(self, array: System.Array, index: int) -> None:
        ...

    def get(self, index: int) -> bool:
        """
        Gets the value of the bit at a specific position in the BitArray.
        
        :param index: The zero-based index of the value to get.
        :returns: The value of the bit at position .
        """
        ...

    def get_enumerator(self) -> System.Collections.IEnumerator:
        """
        Returns an enumerator that iterates through the BitArray.
        
        :returns: An IEnumerator for the entire BitArray.
        """
        ...

    def has_all_set(self) -> bool:
        """
        Determines whether all bits in the BitArray are set to true.
        
        :returns: true if every bit in the BitArray is set to true, or if BitArray is empty; otherwise, false.
        """
        ...

    def has_any_set(self) -> bool:
        """
        Determines whether any bit in the BitArray is set to true.
        
        :returns: true if BitArray is not empty and at least one of its bit is set to true; otherwise, false.
        """
        ...

    def left_shift(self, count: int) -> System.Collections.BitArray:
        """
        Shifts all the bit values of the current BitArray to the left on  bits.
        
        :param count: The number of shifts to make for each bit.
        :returns: The current BitArray.
        """
        ...

    def Not(self) -> System.Collections.BitArray:
        """
        Inverts all the bit values in the current BitArray, so that elements set to true are changed to false,
        and elements set to false are changed to true.
        
        :returns: The current instance with inverted bit values.
        """
        ...

    def Or(self, value: System.Collections.BitArray) -> System.Collections.BitArray:
        """
        Performs the bitwise OR operation between the elements of the current BitArray object and the
        corresponding elements in the specified array. The current BitArray object will be modified to
        store the result of the bitwise OR operation.
        
        :param value: The array with which to perform the bitwise OR operation.
        :returns: An array containing the result of the bitwise OR operation, which is a reference to the current BitArray object.
        """
        ...

    def right_shift(self, count: int) -> System.Collections.BitArray:
        """
        Shifts all the bit values of the current BitArray to the right on  bits.
        
        :param count: The number of shifts to make for each bit.
        :returns: The current BitArray.
        """
        ...

    def set(self, index: int, value: bool) -> None:
        """
        Sets the value of the bit at a specific position in the BitArray.
        
        :param index: The zero-based index of the value to get.
        :param value: The Boolean value to assign to the bit.
        """
        ...

    def set_all(self, value: bool) -> None:
        """
        Sets all bits in the BitArray to the specified value.
        
        :param value: The Boolean value to assign to all bits.
        """
        ...

    def xor(self, value: System.Collections.BitArray) -> System.Collections.BitArray:
        """
        Performs the bitwise XOR operation between the elements of the current BitArray object and the
        corresponding elements in the specified array. The current BitArray object will be modified to
        store the result of the bitwise XOR operation.
        
        :param value: The array with which to perform the bitwise XOR operation.
        :returns: An array containing the result of the bitwise XOR operation, which is a reference to the current BitArray object.
        """
        ...


class IStructuralComparable(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    def compare_to(self, other: typing.Any, comparer: System.Collections.IComparer) -> int:
        ...


class ListDictionaryInternal(System.Object, System.Collections.IDictionary):
    """
    Implements IDictionary using a singly linked list.
    Recommended for collections that typically include fewer than 10 items.
    """

    @property
    def count(self) -> int:
        ...

    @property
    def keys(self) -> System.Collections.ICollection:
        ...

    @property
    def is_read_only(self) -> bool:
        ...

    @property
    def is_fixed_size(self) -> bool:
        ...

    @property
    def is_synchronized(self) -> bool:
        ...

    @property
    def sync_root(self) -> System.Object:
        ...

    @property
    def values(self) -> System.Collections.ICollection:
        ...

    def __getitem__(self, key: typing.Any) -> typing.Any:
        ...

    def __init__(self) -> None:
        ...

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def add(self, key: typing.Any, value: typing.Any) -> None:
        ...

    def clear(self) -> None:
        ...

    def contains(self, key: typing.Any) -> bool:
        ...

    def copy_to(self, array: System.Array, index: int) -> None:
        ...

    def get_enumerator(self) -> System.Collections.IDictionaryEnumerator:
        ...

    def remove(self, key: typing.Any) -> None:
        ...


class IList(System.Collections.ICollection, metaclass=abc.ABCMeta):
    """This class has no documentation."""

    @property
    @abc.abstractmethod
    def is_read_only(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def is_fixed_size(self) -> bool:
        ...

    def __getitem__(self, index: int) -> typing.Any:
        ...

    def __setitem__(self, index: int, value: typing.Any) -> None:
        ...

    def add(self, value: typing.Any) -> int:
        ...

    def clear(self) -> None:
        ...

    def contains(self, value: typing.Any) -> bool:
        ...

    def index_of(self, value: typing.Any) -> int:
        ...

    def insert(self, index: int, value: typing.Any) -> None:
        ...

    def remove(self, value: typing.Any) -> None:
        ...

    def remove_at(self, index: int) -> None:
        ...


class ArrayList(System.Object, System.Collections.IList, System.ICloneable):
    """Implements the IList interface using an array whose size is dynamically increased as required."""

    @property
    def capacity(self) -> int:
        ...

    @capacity.setter
    def capacity(self, value: int) -> None:
        ...

    @property
    def count(self) -> int:
        ...

    @property
    def is_fixed_size(self) -> bool:
        ...

    @property
    def is_read_only(self) -> bool:
        ...

    @property
    def is_synchronized(self) -> bool:
        ...

    @property
    def sync_root(self) -> System.Object:
        ...

    def __getitem__(self, index: int) -> typing.Any:
        ...

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, capacity: int) -> None:
        ...

    @overload
    def __init__(self, c: System.Collections.ICollection) -> None:
        ...

    def __setitem__(self, index: int, value: typing.Any) -> None:
        ...

    @staticmethod
    def adapter(list: System.Collections.IList) -> System.Collections.ArrayList:
        ...

    def add(self, value: typing.Any) -> int:
        ...

    def add_range(self, c: System.Collections.ICollection) -> None:
        ...

    @overload
    def binary_search(self, index: int, count: int, value: typing.Any, comparer: System.Collections.IComparer) -> int:
        ...

    @overload
    def binary_search(self, value: typing.Any) -> int:
        ...

    @overload
    def binary_search(self, value: typing.Any, comparer: System.Collections.IComparer) -> int:
        ...

    def clear(self) -> None:
        ...

    def clone(self) -> System.Object:
        ...

    def contains(self, item: typing.Any) -> bool:
        ...

    @overload
    def copy_to(self, array: System.Array) -> None:
        ...

    @overload
    def copy_to(self, array: System.Array, array_index: int) -> None:
        ...

    @overload
    def copy_to(self, index: int, array: System.Array, array_index: int, count: int) -> None:
        ...

    @staticmethod
    @overload
    def fixed_size(list: System.Collections.IList) -> System.Collections.IList:
        ...

    @staticmethod
    @overload
    def fixed_size(list: System.Collections.ArrayList) -> System.Collections.ArrayList:
        ...

    @overload
    def get_enumerator(self) -> System.Collections.IEnumerator:
        ...

    @overload
    def get_enumerator(self, index: int, count: int) -> System.Collections.IEnumerator:
        ...

    def get_range(self, index: int, count: int) -> System.Collections.ArrayList:
        ...

    @overload
    def index_of(self, value: typing.Any) -> int:
        ...

    @overload
    def index_of(self, value: typing.Any, start_index: int) -> int:
        ...

    @overload
    def index_of(self, value: typing.Any, start_index: int, count: int) -> int:
        ...

    def insert(self, index: int, value: typing.Any) -> None:
        ...

    def insert_range(self, index: int, c: System.Collections.ICollection) -> None:
        ...

    @overload
    def last_index_of(self, value: typing.Any) -> int:
        ...

    @overload
    def last_index_of(self, value: typing.Any, start_index: int) -> int:
        ...

    @overload
    def last_index_of(self, value: typing.Any, start_index: int, count: int) -> int:
        ...

    @staticmethod
    @overload
    def read_only(list: System.Collections.IList) -> System.Collections.IList:
        ...

    @staticmethod
    @overload
    def read_only(list: System.Collections.ArrayList) -> System.Collections.ArrayList:
        ...

    def remove(self, obj: typing.Any) -> None:
        ...

    def remove_at(self, index: int) -> None:
        ...

    def remove_range(self, index: int, count: int) -> None:
        ...

    @staticmethod
    def repeat(value: typing.Any, count: int) -> System.Collections.ArrayList:
        ...

    @overload
    def reverse(self) -> None:
        ...

    @overload
    def reverse(self, index: int, count: int) -> None:
        ...

    def set_range(self, index: int, c: System.Collections.ICollection) -> None:
        ...

    @overload
    def sort(self) -> None:
        ...

    @overload
    def sort(self, comparer: System.Collections.IComparer) -> None:
        ...

    @overload
    def sort(self, index: int, count: int, comparer: System.Collections.IComparer) -> None:
        ...

    @staticmethod
    @overload
    def synchronized(list: System.Collections.IList) -> System.Collections.IList:
        ...

    @staticmethod
    @overload
    def synchronized(list: System.Collections.ArrayList) -> System.Collections.ArrayList:
        ...

    @overload
    def to_array(self) -> typing.List[System.Object]:
        ...

    @overload
    def to_array(self, type: typing.Type) -> System.Array:
        ...

    def trim_to_size(self) -> None:
        ...


class Comparer(System.Object, System.Collections.IComparer, System.Runtime.Serialization.ISerializable):
    """Compares two objects for equivalence, where string comparisons are case-sensitive."""

    DEFAULT: System.Collections.Comparer = ...

    DEFAULT_INVARIANT: System.Collections.Comparer = ...

    def __init__(self, culture: System.Globalization.CultureInfo) -> None:
        ...

    def compare(self, a: typing.Any, b: typing.Any) -> int:
        ...

    def get_object_data(self, info: System.Runtime.Serialization.SerializationInfo, context: System.Runtime.Serialization.StreamingContext) -> None:
        """Obsoletions.LegacyFormatterImplMessage"""
        warnings.warn("Obsoletions.LegacyFormatterImplMessage", DeprecationWarning)


class IStructuralEquatable(metaclass=abc.ABCMeta):
    """This class has no documentation."""

    def equals(self, other: typing.Any, comparer: System.Collections.IEqualityComparer) -> bool:
        ...

    def get_hash_code(self, comparer: System.Collections.IEqualityComparer) -> int:
        ...


class StructuralComparisons(System.Object):
    """This class has no documentation."""

    STRUCTURAL_COMPARER: System.Collections.IComparer

    STRUCTURAL_EQUALITY_COMPARER: System.Collections.IEqualityComparer


