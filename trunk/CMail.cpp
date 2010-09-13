////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#include "stdafx.h"

CMail::CMail()
{
	m_smtpServer = "mail.konradrodzik.pl";
	m_from = "ray.tracer.benchmark@konradrodzik.pl";
	m_password = "ray.tracer.benchmark.password";
	m_to = "konrad.rodzik@gmail.com";

	m_mail.setserver(m_smtpServer);
	m_mail.username(m_from);
	m_mail.password(m_password);
	m_mail.setsender(m_from);
	m_mail.addrecipient(m_to);
}

void CMail::setSubject( const char* title )
{
	m_subject = title;
	m_mail.setsubject(m_subject);
}

void CMail::setBody( const char* body )
{
	m_body = body;
	m_mail.setmessage(m_body);
}

void CMail::addAttachment( const char* attach_file )
{
	m_mail.attach(attach_file);
}

const std::string CMail::send()
{
	// Send mail and return response
	m_mail.send();
	return m_mail.response();

}