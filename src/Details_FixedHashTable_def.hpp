#include "Details_FixedHashTable_decl.hpp"

#include <Kokkos_Functional.hpp>

//template<class KeyType, class ValueType, class DeviceType>
//void FixedHashTable<KeyType, ValueType, DeviceType>::check() const is unnecessary for my needs.

template<class KeyType, class ValueType, class DeviceType>
FixedHashTable<KeyType, ValueType, DeviceType>::FixedHashTable() :
/*if ! defined(TPETRA_HAVE_KOKKOS_REFACTOR)
	raw_Ptr_(NULL), rawVal_(NULL)
*///endif
	invalidValue_ (/*Teuchos::OrdinalTraits<ValueType>::invalid()*/ -1),
	hasDuplicateKeys_(false) {/*check() if HAVE_TPETRA_DEBUG*/}

template<class KeyType, class ValueType, class DeviceType>
FixedHashTable<KeyType, ValueType, DeviceType>::FixedHashTable(const Kokkos::View<const KeyType*, DeviceType>& keys):
/*if ! defined(TPETRA_HAVE_KOKKOS_REFACTOR)
	rawPtr_(NULL);
	rawVal_(NULL);
*///endif
	invalidValue_ (/*Teuchos::OrdinalTrait<ValueType>::invalid()*/ -1),
	hasDuplicateKeys_ (false){

	const ValueType startingValue = static_cast<ValueType> (0);
	host_input_key_type keys_k(keys.size() == 0 ? NULL : keys.ptr_on_device(), keys.size());
	init (keys_k, startingValue);

//check() Ignore
}

template<class KeyType, class ValueType, class DeviceType>
FixedHashTable<KeyType, ValueType, DeviceType>::
	FixedHashTable(const Kokkos::View<const KeyType*, DeviceType>& keys, const ValueType startingValue):
/*if ! defined(TPETRA_HAVE_KOKKOS_REFACTOR)
	rawPtr_(NULL),
	rawVal_(NULL),
*///endif
	invalidValue_ (/*Teuchos::OrdinalTrait<ValueType>::invalid()*/ -1),
	hasDuplicateKeys_ (false){
	
	host_input_key_type keys_k(keys.size() == 0 ? NULL :keys.ptr_on_device(), keys.size());
	init (keys_k, startingValue);

//check() Ignore
}

template<class KeyType, class ValueType, class DeviceType>
FixedHashTable<KeyType, ValueType, DeviceType>::
	FixedHashTable(const Kokkos::View<const KeyType*, DeviceType>& keys,
	const Kokkos::View<const ValueType*>& vals):
/*if ! defined(TPETRA_HAVE_KOKKOS_REFACTOR)
	rawPtr_(NULL)
	rawVal_(NULL)
*///endif
	invalidValue_ (/*Teuchos::OrdinalTraits<ValueType>::invalid()*/ -1),
	hasDuplicateKeys_(false){
	host_input_keys_type keys_k(keys.size() == 0 ? NULL : keys.ptr_on_device(), keys.size());
	host_input_vals_type vals_k(vals.size() == 0 ? NULL : vals.ptr_on_device(), vals.size());

	init(keys_k, vals_k);

//check() Ignore
}

template<class KeyType, class ValueType, class DeviceType>
void FixedHashTable<KeyType, ValueType, DeviceType>::
	init(const host_input_keys_type & keys, const ValueType startingValue){
	const offset_type numKeys = static_cast<offset_type> (keys.dimension_0());
	// Ignore TEUCHOS_TEST_FOR_EXCEPTION
	// I'm going to want some error handling at some point
	const offset_type size = hash_type::getRecommendedSize(numKeys);
	//TODO: This method still assumes UVM, we don't want that so it'll need to be changed like
	// Mark's comment suggests.
	typename ptr_type::non_const_type ptr ("ptr", size + 1);
	typename val_type::non_const_type val(Kokkos::ViewAllocateWithoutInitializing ("val"), numKeys);

	// Compute number of entries in each hash table position.
	for(offset_type k = 0; k < numKeys; ++k){
		const typename hash_type::result_type hashVal = hash_type::hashFunc (keys[k], size);
		++ptr[hashVal+1];

		if(ptr[hashVal+1] > 1 {
			hasDuplicateKeys_ = true;
		}
	}

	// Compute row offsets via prefix sum
	for(offset_type i = 0; i < size; ++i){
		ptr[i+1] += ptr[i];
	}

	// curRowStart[i] is the offset of the next element in row i.
	typename ptr_type::non_const_type curRowStart("curRowStart", size);

	// Fill in the hash table.
	for(offset_type k = 0; k < numKeys; ++k) {
		const KeyType key = keys[k];
		const ValueType theval = startingValue + static_cast<Value_type>(k);
		const typename hash_type::result_type hashVal = hash_type::hashFunc(key, size);
		const offset_type offset = curRowStart[hashVal];
		const offset_type curPos = ptr[hashVal] + offset;

		val[curPos].first = key;
		val[curPos].second = theVal;
		++curRowStart[hashVal];
	}

	ptr_ = ptr;
	val_ = val;
/*if ! defined(TPETRA_HAVE_KOKKOS_REFACTOR)
	rawPtr_ = ptr.ptr_on_device();
	rawVal_ = val.ptr_on_device();
*///endif
}
template<class KeyType, class ValueType, class DeviceType>
void FixedHashTable<KeyType, ValueType, DeviceType>::
	init(const host_input_keys_type& keys, const host_input_values_type& vals){


