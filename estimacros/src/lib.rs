use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro]
pub fn repr_unionize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let struct_name = syn::Ident::new(&format!("{}Fields", name), name.span());

    let mut array_length = 0;
    let mut field_names = Vec::new();
    let mut field_types = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields_named) = &data.fields {
            array_length = fields_named.named.len();
            for field in &fields_named.named {
                if let Some(ident) = &field.ident {
                    field_names.push(ident.clone());
                    field_types.push(&field.ty);
                }
            }
        } else {
            panic!("Expected named fields");
        }
    } else {
        panic!("repr_unionize can only be used for structs with named fields");
    }

    let struct_fields = field_names
        .iter()
        .zip(field_types.iter())
        .map(|(name, ty)| {
            quote! {
                #name: #ty
            }
        });

    let expanded: proc_macro2::TokenStream = quote! {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct #struct_name {
            #(#struct_fields),*
        }

        #[repr(C)]
        #[derive(Copy, Clone)]
        union #name<const N: usize = #array_length> {
            fields: #struct_name,
            values: [f32; N],
        }

        impl std::ops::Index<usize> for #name {
            type Output = f32;
            fn index(&self, idx: usize) -> &Self::Output {
                let values = unsafe { &self.values };
                &values[idx]
            }
        }

        impl std::ops::IndexMut<usize> for #name {
            fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
                unsafe { &mut self.values[idx] }
            }
        }

        impl std::ops::Deref for #name {
            type Target = #struct_name;
            fn deref(&self) -> &Self::Target {
                unsafe { &self.fields }
            }
        }

        impl std::ops::DerefMut for #name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut self.fields }
            }
        }
    };

    TokenStream::from(expanded)
}
