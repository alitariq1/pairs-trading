from typing import overload
from enum import Enum
import typing

import System.Numerics
import System.Runtime.Intrinsics

System_Runtime_Intrinsics_Vector128 = typing.Any
T = typing.Any
System_Runtime_Intrinsics_Vector256 = typing.Any
System_Runtime_Intrinsics_Vector64 = typing.Any
System_Runtime_Intrinsics_Vector512 = typing.Any

System_Runtime_Intrinsics_Vector128_T = typing.TypeVar("System_Runtime_Intrinsics_Vector128_T")
System_Runtime_Intrinsics_Vector256_T = typing.TypeVar("System_Runtime_Intrinsics_Vector256_T")
System_Runtime_Intrinsics_Vector64_T = typing.TypeVar("System_Runtime_Intrinsics_Vector64_T")
System_Runtime_Intrinsics_Vector512_T = typing.TypeVar("System_Runtime_Intrinsics_Vector512_T")


class Vector128(typing.Generic[System_Runtime_Intrinsics_Vector128_T], System.Runtime.Intrinsics.ISimdVector[System_Runtime_Intrinsics_Vector128, System_Runtime_Intrinsics_Vector128_T]):
    """Represents a 128-bit vector of a specified numeric type that is suitable for low-level optimization of parallel algorithms."""

    ALL_BITS_SET: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]
    """Gets a new Vector128{T} with all bits set to 1."""

    COUNT: int
    """Gets the number of T that are in a Vector128{T}."""

    INDICES: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]
    """Gets a new Vector128{T} with the elements set to their index."""

    IS_SUPPORTED: bool
    """Gets true if T is supported; otherwise, false."""

    ONE: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]
    """Gets a new Vector128{T} with all elements initialized to one."""

    ZERO: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]
    """Gets a new Vector128{T} with all elements initialized to zero."""

    IS_HARDWARE_ACCELERATED: bool
    """Gets a value that indicates whether 128-bit vector operations are subject to hardware acceleration through JIT intrinsic support."""

    E: System.Runtime.Intrinsics.Vector128[T]
    """Gets a new vector with all elements initialized to IFloatingPointConstants{TSelf}.E."""

    PI: System.Runtime.Intrinsics.Vector128[T]
    """Gets a new vector with all elements initialized to IFloatingPointConstants{TSelf}.Pi."""

    TAU: System.Runtime.Intrinsics.Vector128[T]
    """Gets a new vector with all elements initialized to IFloatingPointConstants{TSelf}.Tau."""

    @overload
    def __add__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __add__(self) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __and__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __eq__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> bool:
        """
        Compares two vectors to determine if all elements are equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if all elements in  were equal to the corresponding element in .
        """
        ...

    def __getitem__(self, index: int) -> System_Runtime_Intrinsics_Vector128_T:
        """
        Gets the element at the specified index.
        
        :param index: The index of the element to get.
        :returns: The value of the element at .
        """
        ...

    @overload
    def __iadd__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __iadd__(self) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __iand__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __ilshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System_Runtime_Intrinsics_Vector128_T) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __invert__(self) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the ones-complement of a vector.
        
        :param vector: The vector whose ones-complement is to be computed.
        :returns: A vector whose elements are the ones-complement of the corresponding elements in .
        """
        ...

    def __ior__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __irshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __isub__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __isub__(self) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __itruediv__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __itruediv__(self, right: System_Runtime_Intrinsics_Vector128_T) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __ixor__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    def __lshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System_Runtime_Intrinsics_Vector128_T) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __ne__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> bool:
        """
        Compares two vectors to determine if any elements are not equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if any elements in  was not equal to the corresponding element in .
        """
        ...

    def __or__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __rshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __sub__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __sub__(self) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __truediv__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __truediv__(self, right: System_Runtime_Intrinsics_Vector128_T) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __xor__(self, right: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    @staticmethod
    def as_plane(value: System.Runtime.Intrinsics.Vector128[float]) -> System.Numerics.Plane:
        """
        Reinterprets a Vector128<Single> as a new Plane.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Plane.
        """
        ...

    @staticmethod
    def as_quaternion(value: System.Runtime.Intrinsics.Vector128[float]) -> System.Numerics.Quaternion:
        """
        Reinterprets a Vector128<Single> as a new Quaternion.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Quaternion.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128(value: System.Numerics.Plane) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Plane as a new Vector128<Single>.
        
        :param value: The plane to reinterpret.
        :returns: reinterpreted as a new Vector128<Single>.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128(value: System.Numerics.Quaternion) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Quaternion as a new Vector128<Single>.
        
        :param value: The quaternion to reinterpret.
        :returns: reinterpreted as a new Vector128<Single>.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128(value: System.Numerics.Vector2) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Vector2 as a new Vector128<Single> with the new elements zeroed.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector128<Single> with the new elements zeroed.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128(value: System.Numerics.Vector3) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Vector3 as a new Vector128<Single> with the new elements zeroed.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector128<Single> with the new elements zeroed.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128(value: System.Numerics.Vector4) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Vector4 as a new Vector128<Single>.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector128<Single>.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128_unsafe(value: System.Numerics.Vector2) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Vector2 as a new Vector128<Single>, leaving the new elements undefined.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector128<Single>.
        """
        ...

    @staticmethod
    @overload
    def as_vector_128_unsafe(value: System.Numerics.Vector3) -> System.Runtime.Intrinsics.Vector128[float]:
        """
        Reinterprets a Vector3 as a new Vector128<Single>, leaving the new elements undefined.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector128<Single>.
        """
        ...

    @staticmethod
    def as_vector_2(value: System.Runtime.Intrinsics.Vector128[float]) -> System.Numerics.Vector2:
        """
        Reinterprets a Vector128<Single> as a new Vector2.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector2.
        """
        ...

    @staticmethod
    def as_vector_3(value: System.Runtime.Intrinsics.Vector128[float]) -> System.Numerics.Vector3:
        """
        Reinterprets a Vector128<Single> as a new Vector3.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector3.
        """
        ...

    @staticmethod
    def as_vector_4(value: System.Runtime.Intrinsics.Vector128[float]) -> System.Numerics.Vector4:
        """
        Reinterprets a Vector128<Single> as a new Vector4.
        
        :param value: The vector to reinterpret.
        :returns: reinterpreted as a new Vector4.
        """
        ...

    @overload
    def equals(self, obj: typing.Any) -> bool:
        """
        Determines whether the specified object is equal to the current instance.
        
        :param obj: The object to compare with the current instance.
        :returns: true if  is a Vector128{T} and is equal to the current instance; otherwise, false.
        """
        ...

    @overload
    def equals(self, other: System.Runtime.Intrinsics.Vector128[System_Runtime_Intrinsics_Vector128_T]) -> bool:
        """
        Determines whether the specified Vector128{T} is equal to the current instance.
        
        :param other: The Vector128{T} to compare with the current instance.
        :returns: true if  is equal to the current instance; otherwise, false.
        """
        ...

    def get_hash_code(self) -> int:
        """
        Gets the hash code for the instance.
        
        :returns: The hash code for the instance.
        """
        ...

    def to_string(self) -> str:
        """
        Converts the current instance to an equivalent string representation.
        
        :returns: An equivalent string representation of the current instance.
        """
        ...


class Vector256(typing.Generic[System_Runtime_Intrinsics_Vector256_T], System.Runtime.Intrinsics.ISimdVector[System_Runtime_Intrinsics_Vector256, System_Runtime_Intrinsics_Vector256_T]):
    """Represents a 256-bit vector of a specified numeric type that is suitable for low-level optimization of parallel algorithms."""

    ALL_BITS_SET: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]
    """Gets a new Vector256{T} with all bits set to 1."""

    COUNT: int
    """Gets the number of T that are in a Vector256{T}."""

    INDICES: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]
    """Gets a new Vector256{T} with the elements set to their index."""

    IS_SUPPORTED: bool
    """Gets true if T is supported; otherwise, false."""

    ONE: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]
    """Gets a new Vector256{T} with all elements initialized to one."""

    ZERO: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]
    """Gets a new Vector256{T} with all elements initialized to zero."""

    IS_HARDWARE_ACCELERATED: bool
    """Gets a value that indicates whether 256-bit vector operations are subject to hardware acceleration through JIT intrinsic support."""

    E: System.Runtime.Intrinsics.Vector256[T]

    PI: System.Runtime.Intrinsics.Vector256[T]

    TAU: System.Runtime.Intrinsics.Vector256[T]

    @overload
    def __add__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __add__(self) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __and__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __eq__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> bool:
        """
        Compares two vectors to determine if all elements are equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if all elements in  were equal to the corresponding element in .
        """
        ...

    def __getitem__(self, index: int) -> System_Runtime_Intrinsics_Vector256_T:
        """
        Gets the element at the specified index.
        
        :param index: The index of the element to get.
        :returns: The value of the element at .
        """
        ...

    @overload
    def __iadd__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __iadd__(self) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __iand__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __ilshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System_Runtime_Intrinsics_Vector256_T) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __invert__(self) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the ones-complement of a vector.
        
        :param vector: The vector whose ones-complement is to be computed.
        :returns: A vector whose elements are the ones-complement of the corresponding elements in .
        """
        ...

    def __ior__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __irshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __isub__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __isub__(self) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __itruediv__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __itruediv__(self, right: System_Runtime_Intrinsics_Vector256_T) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __ixor__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    def __lshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System_Runtime_Intrinsics_Vector256_T) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __ne__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> bool:
        """
        Compares two vectors to determine if any elements are not equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if any elements in  was not equal to the corresponding element in .
        """
        ...

    def __or__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __rshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __sub__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __sub__(self) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __truediv__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __truediv__(self, right: System_Runtime_Intrinsics_Vector256_T) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __xor__(self, right: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    @overload
    def equals(self, obj: typing.Any) -> bool:
        """
        Determines whether the specified object is equal to the current instance.
        
        :param obj: The object to compare with the current instance.
        :returns: true if  is a Vector256{T} and is equal to the current instance; otherwise, false.
        """
        ...

    @overload
    def equals(self, other: System.Runtime.Intrinsics.Vector256[System_Runtime_Intrinsics_Vector256_T]) -> bool:
        """
        Determines whether the specified Vector256{T} is equal to the current instance.
        
        :param other: The Vector256{T} to compare with the current instance.
        :returns: true if  is equal to the current instance; otherwise, false.
        """
        ...

    def get_hash_code(self) -> int:
        """
        Gets the hash code for the instance.
        
        :returns: The hash code for the instance.
        """
        ...

    def to_string(self) -> str:
        """
        Converts the current instance to an equivalent string representation.
        
        :returns: An equivalent string representation of the current instance.
        """
        ...


class Vector64(typing.Generic[System_Runtime_Intrinsics_Vector64_T], System.Runtime.Intrinsics.ISimdVector[System_Runtime_Intrinsics_Vector64, System_Runtime_Intrinsics_Vector64_T]):
    """Represents a 64-bit vector of a specified numeric type that is suitable for low-level optimization of parallel algorithms."""

    IS_HARDWARE_ACCELERATED: bool
    """Gets a value that indicates whether 64-bit vector operations are subject to hardware acceleration through JIT intrinsic support."""

    E: System.Runtime.Intrinsics.Vector64[T]

    PI: System.Runtime.Intrinsics.Vector64[T]

    TAU: System.Runtime.Intrinsics.Vector64[T]

    ALL_BITS_SET: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]
    """Gets a new Vector64{T} with all bits set to 1."""

    COUNT: int
    """Gets the number of T that are in a Vector64{T}."""

    INDICES: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]
    """Gets a new Vector64{T} with the elements set to their index."""

    IS_SUPPORTED: bool
    """Gets true if T is supported; otherwise, false."""

    ONE: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]
    """Gets a new Vector64{T} with all elements initialized to one."""

    ZERO: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]
    """Gets a new Vector64{T} with all elements initialized to zero."""

    @overload
    def __add__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __add__(self) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __and__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __eq__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> bool:
        """
        Compares two vectors to determine if all elements are equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if all elements in  were equal to the corresponding element in .
        """
        ...

    def __getitem__(self, index: int) -> System_Runtime_Intrinsics_Vector64_T:
        """
        Gets the element at the specified index.
        
        :param index: The index of the element to get.
        :returns: The value of the element at .
        """
        ...

    @overload
    def __iadd__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __iadd__(self) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __iand__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __ilshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System_Runtime_Intrinsics_Vector64_T) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __invert__(self) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the ones-complement of a vector.
        
        :param vector: The vector whose ones-complement is to be computed.
        :returns: A vector whose elements are the ones-complement of the corresponding elements in .
        """
        ...

    def __ior__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __irshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __isub__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __isub__(self) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __itruediv__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __itruediv__(self, right: System_Runtime_Intrinsics_Vector64_T) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __ixor__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    def __lshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System_Runtime_Intrinsics_Vector64_T) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __ne__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> bool:
        """
        Compares two vectors to determine if any elements are not equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if any elements in  was not equal to the corresponding element in .
        """
        ...

    def __or__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __rshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __sub__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __sub__(self) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __truediv__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __truediv__(self, right: System_Runtime_Intrinsics_Vector64_T) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __xor__(self, right: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    @overload
    def equals(self, obj: typing.Any) -> bool:
        """
        Determines whether the specified object is equal to the current instance.
        
        :param obj: The object to compare with the current instance.
        :returns: true if  is a Vector64{T} and is equal to the current instance; otherwise, false.
        """
        ...

    @overload
    def equals(self, other: System.Runtime.Intrinsics.Vector64[System_Runtime_Intrinsics_Vector64_T]) -> bool:
        """
        Determines whether the specified Vector64{T} is equal to the current instance.
        
        :param other: The Vector64{T} to compare with the current instance.
        :returns: true if  is equal to the current instance; otherwise, false.
        """
        ...

    def get_hash_code(self) -> int:
        """
        Gets the hash code for the instance.
        
        :returns: The hash code for the instance.
        """
        ...

    def to_string(self) -> str:
        """
        Converts the current instance to an equivalent string representation.
        
        :returns: An equivalent string representation of the current instance.
        """
        ...


class Vector512(typing.Generic[System_Runtime_Intrinsics_Vector512_T], System.Runtime.Intrinsics.ISimdVector[System_Runtime_Intrinsics_Vector512, System_Runtime_Intrinsics_Vector512_T]):
    """Represents a 512-bit vector of a specified numeric type that is suitable for low-level optimization of parallel algorithms."""

    ALL_BITS_SET: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]
    """Gets a new Vector512{T} with all bits set to 1."""

    COUNT: int
    """Gets the number of T that are in a Vector512{T}."""

    INDICES: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]
    """Gets a new Vector512{T} with the elements set to their index."""

    IS_SUPPORTED: bool
    """Gets true if T is supported; otherwise, false."""

    ONE: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]
    """Gets a new Vector512{T} with all elements initialized to one."""

    ZERO: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]
    """Gets a new Vector512{T} with all elements initialized to zero."""

    IS_HARDWARE_ACCELERATED: bool
    """Gets a value that indicates whether 512-bit vector operations are subject to hardware acceleration through JIT intrinsic support."""

    E: System.Runtime.Intrinsics.Vector512[T]

    PI: System.Runtime.Intrinsics.Vector512[T]

    TAU: System.Runtime.Intrinsics.Vector512[T]

    @overload
    def __add__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __add__(self) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __and__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __eq__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> bool:
        """
        Compares two vectors to determine if all elements are equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if all elements in  were equal to the corresponding element in .
        """
        ...

    def __getitem__(self, index: int) -> System_Runtime_Intrinsics_Vector512_T:
        """
        Gets the element at the specified index.
        
        :param index: The index of the element to get.
        :returns: The value of the element at .
        """
        ...

    @overload
    def __iadd__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Adds two vectors to compute their sum.
        
        :param left: The vector to add with .
        :param right: The vector to add with .
        :returns: The sum of  and .
        """
        ...

    @overload
    def __iadd__(self) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Returns a given vector unchanged.
        
        :param value: The vector.
        """
        ...

    def __iand__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the bitwise-and of two vectors.
        
        :param left: The vector to bitwise-and with .
        :param right: The vector to bitwise-and with .
        :returns: The bitwise-and of  and .
        """
        ...

    def __ilshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System_Runtime_Intrinsics_Vector512_T) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __imul__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __invert__(self) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the ones-complement of a vector.
        
        :param vector: The vector whose ones-complement is to be computed.
        :returns: A vector whose elements are the ones-complement of the corresponding elements in .
        """
        ...

    def __ior__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __irshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __isub__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __isub__(self) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __itruediv__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __itruediv__(self, right: System_Runtime_Intrinsics_Vector512_T) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __ixor__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    def __lshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Shifts each element of a vector left by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted left by .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies two vectors to compute their element-wise product.
        
        :param left: The vector to multiply with .
        :param right: The vector to multiply with .
        :returns: The element-wise product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System_Runtime_Intrinsics_Vector512_T) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The vector to multiply with .
        :param right: The scalar to multiply with .
        :returns: The product of  and .
        """
        ...

    @overload
    def __mul__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Multiplies a vector by a scalar to compute their product.
        
        :param left: The scalar to multiply with .
        :param right: The vector to multiply with .
        :returns: The product of  and .
        """
        ...

    def __ne__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> bool:
        """
        Compares two vectors to determine if any elements are not equal.
        
        :param left: The vector to compare with .
        :param right: The vector to compare with .
        :returns: true if any elements in  was not equal to the corresponding element in .
        """
        ...

    def __or__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the bitwise-or of two vectors.
        
        :param left: The vector to bitwise-or with .
        :param right: The vector to bitwise-or with .
        :returns: The bitwise-or of  and .
        """
        ...

    def __rshift__(self, shift_count: int) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Shifts (signed) each element of a vector right by the specified amount.
        
        :param value: The vector whose elements are to be shifted.
        :param shift_count: The number of bits by which to shift each element.
        :returns: A vector whose elements where shifted right by .
        """
        ...

    @overload
    def __sub__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Subtracts two vectors to compute their difference.
        
        :param left: The vector from which  will be subtracted.
        :param right: The vector to subtract from .
        :returns: The difference of  and .
        """
        ...

    @overload
    def __sub__(self) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the unary negation of a vector.
        
        :param vector: The vector to negate.
        :returns: A vector whose elements are the unary negation of the corresponding elements in .
        """
        ...

    @overload
    def __truediv__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Divides two vectors to compute their quotient.
        
        :param left: The vector that will be divided by .
        :param right: The vector that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    @overload
    def __truediv__(self, right: System_Runtime_Intrinsics_Vector512_T) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Divides a vector by a scalar to compute the per-element quotient.
        
        :param left: The vector that will be divided by .
        :param right: The scalar that will divide .
        :returns: The quotient of  divided by .
        """
        ...

    def __xor__(self, right: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]:
        """
        Computes the exclusive-or of two vectors.
        
        :param left: The vector to exclusive-or with .
        :param right: The vector to exclusive-or with .
        :returns: The exclusive-or of  and .
        """
        ...

    @overload
    def equals(self, obj: typing.Any) -> bool:
        """
        Determines whether the specified object is equal to the current instance.
        
        :param obj: The object to compare with the current instance.
        :returns: true if  is a Vector512{T} and is equal to the current instance; otherwise, false.
        """
        ...

    @overload
    def equals(self, other: System.Runtime.Intrinsics.Vector512[System_Runtime_Intrinsics_Vector512_T]) -> bool:
        """
        Determines whether the specified Vector512{T} is equal to the current instance.
        
        :param other: The Vector512{T} to compare with the current instance.
        :returns: true if  is equal to the current instance; otherwise, false.
        """
        ...

    def get_hash_code(self) -> int:
        """
        Gets the hash code for the instance.
        
        :returns: The hash code for the instance.
        """
        ...

    def to_string(self) -> str:
        """
        Converts the current instance to an equivalent string representation.
        
        :returns: An equivalent string representation of the current instance.
        """
        ...


