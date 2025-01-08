#[macro_export]
macro_rules! vector_union {
    (
        $name:ident, $type:ty, $fields_name:ident {
            $($field:ident $( : $nested_type:ident )?),*
        }
    ) => {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub union $name {
            pub values: [$type; vector_union!(@count $type, $($field $( : $nested_type)?),*)],
            pub fields: $fields_name,
        }

        impl $name {
            const SIZE: usize = vector_union!(@count $type, $($field $( : $nested_type)?),*);
        }

        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $fields_name {
            $(pub $field: vector_union!(@field_type $type, $($nested_type)?),)*
        }

        impl std::ops::Index<usize> for $name {
            type Output = f32;
            fn index(&self, idx: usize) -> &Self::Output {
                let values = unsafe { &self.values };
                &values[idx]
            }
        }

        impl std::ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
                unsafe { &mut self.values[idx] }
            }
        }

        impl std::ops::Deref for $name {
            type Target = $fields_name;
            fn deref(&self) -> &Self::Target {
                unsafe { &self.fields }
            }
        }

        impl std::ops::DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut self.fields }
            }
        }
    };

    (@count $type:ty, $($field:ident $( : $nested_type:ident )?),*) => {
        0 $(+ vector_union!(@field_size $type, $($nested_type)?))*
    };

    (@field_size $type:ty, $nested_type:ident) => {
        { $nested_type::SIZE }
    };

    (@field_size $type:ty,) => {
        1
    };

    (@field_type $type:ty, $nested_type:ident) => { $nested_type };
    (@field_type $type:ty,) => { $type };
}
